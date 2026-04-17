"""
app.py — Flask backend for the CCTV Illegal Dumping Monitoring Dashboard.

Routes:
  GET  /                        → index.html
  GET  /api/cameras             → list cameras
  POST /api/cameras             → add camera {name, source, latitude?, longitude?, settings?}
  PATCH /api/cameras/<id>       → update camera metadata {name?, latitude?, longitude?, settings?}
  DELETE /api/cameras/<id>      → remove camera
  POST /api/cameras/<id>/start  → start pipeline
  POST /api/cameras/<id>/stop   → stop pipeline
  GET  /video/<id>              → MJPEG stream
  GET  /events                  → SSE push stream (trash events)
  GET  /api/events              → last N logged events (JSON)
  GET  /snapshots/<filename>    → serve snapshot image

Run:
    python app.py --model best_model.pth --threshold 0.5
"""

import argparse
import json
import os
import queue
import threading
import time
import uuid
from typing import Any, Dict

import collections

import torch
from flask import Flask, Response, jsonify, request, send_from_directory

from core.classifier import load_model
from core.pipeline import CameraPipeline

# Limit PyTorch CPU threads to avoid starving the OS, Flask, and OpenCV.
torch.set_num_threads(2)

# ─────────────────────────────────────────────────────────────────────────────
# Directories
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT_DIR = os.path.join(BASE_DIR, "snapshots")
LOG_DIR      = os.path.join(BASE_DIR, "logs")
LOG_FILE     = os.path.join(LOG_DIR, "events.jsonl")

os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=BASE_DIR)

_model = None
_class_names = None
_device = None

_settings: Dict[str, Any] = {
    "bg_history":            500,
    "bg_var_threshold":      50,
    "min_area":              500,
    "max_distance":          50.0,
    "max_missed":            15,
    "stationary_distance":   2.0,
    "stationary_min_frames": 30,
    "threshold":             0.5,
    "jpeg_quality":          70,
}
_settings_lock = threading.Lock()

# Shared event queue — CameraPipelines push here; SSE generator reads from it.
# SSE clients each get their own per-connection queue fed by the broadcaster.
_raw_event_queue: queue.Queue = queue.Queue(maxsize=200)

# Per-connection SSE queues
_sse_clients: Dict[str, queue.Queue] = {}
_sse_lock = threading.Lock()

# Camera registry
_cameras: Dict[str, CameraPipeline] = {}
_cameras_lock = threading.Lock()

# Per-camera settings overrides (cam_id → partial settings dict)
_camera_settings: Dict[str, Dict[str, Any]] = {}

# In-memory ring buffer of recent events for fast /api/events serving
_recent_events: collections.deque = collections.deque(maxlen=500)


# ─────────────────────────────────────────────────────────────────────────────
# SSE broadcaster thread — fans out events from the raw queue to all clients
# ─────────────────────────────────────────────────────────────────────────────

def _broadcaster() -> None:
    while True:
        try:
            event = _raw_event_queue.get(timeout=1)
        except queue.Empty:
            continue

        # Persist to log file
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(event) + "\n")

        # Keep in ring buffer for fast API access
        _recent_events.appendleft(event)

        # Fan out to all connected SSE clients
        with _sse_lock:
            dead = []
            for cid, q in _sse_clients.items():
                try:
                    q.put_nowait(event)
                except queue.Full:
                    dead.append(cid)
            for cid in dead:
                del _sse_clients[cid]


threading.Thread(target=_broadcaster, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _camera_info(cam: CameraPipeline) -> Dict[str, Any]:
    cam_id = cam.camera_id
    overrides = _camera_settings.get(cam_id, {})
    return {
        "id":              cam.camera_id,
        "name":            cam.name,
        "source":          str(cam.source),
        "status":          cam.status,
        "error":           cam.error_msg,
        "confirmed_trash": cam.confirmed_count,
        "total_drops":     cam.total_drops,
        "latitude":        cam.latitude,
        "longitude":       cam.longitude,
        "settings":        overrides,
    }


def _require_model() -> bool:
    return _model is not None


# ─────────────────────────────────────────────────────────────────────────────
# Routes — settings
# ─────────────────────────────────────────────────────────────────────────────

_SETTINGS_SCHEMA: Dict[str, tuple] = {
    "bg_history":            (int,   50,    5000),
    "bg_var_threshold":      (int,   1,     500),
    "min_area":              (int,   50,    50000),
    "max_distance":          (float, 1.0,   500.0),
    "max_missed":            (int,   1,     120),
    "stationary_distance":   (float, 0.1,   50.0),
    "stationary_min_frames": (int,   1,     300),
    "threshold":             (float, 0.01,  1.0),
    "jpeg_quality":          (int,   10,    100),
}


@app.route("/api/settings", methods=["GET"])
def get_settings():
    with _settings_lock:
        return jsonify(dict(_settings))


@app.route("/api/settings", methods=["POST"])
def update_settings():
    data = request.get_json(silent=True) or {}
    errors: Dict[str, str] = {}
    updates: Dict[str, Any] = {}

    for key, (typ, lo, hi) in _SETTINGS_SCHEMA.items():
        if key not in data:
            continue
        try:
            val = typ(data[key])
        except (ValueError, TypeError):
            errors[key] = f"must be {typ.__name__}"
            continue
        if not (lo <= val <= hi):
            errors[key] = f"must be between {lo} and {hi}"
            continue
        updates[key] = val

    if errors:
        return jsonify({"errors": errors}), 400

    with _settings_lock:
        _settings.update(updates)
        with _cameras_lock:
            for cam in _cameras.values():
                if "threshold" in updates:
                    cam.threshold = updates["threshold"]
                if "jpeg_quality" in updates:
                    cam.jpeg_quality = updates["jpeg_quality"]

    return jsonify(dict(_settings))


# ─────────────────────────────────────────────────────────────────────────────
# Routes — pages
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(os.path.join(BASE_DIR, "static"), "index.html")


@app.route("/snapshots/<path:filename>")
def snapshot(filename):
    return send_from_directory(SNAPSHOT_DIR, filename)


# ─────────────────────────────────────────────────────────────────────────────
# Routes — camera management
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/cameras", methods=["GET"])
def list_cameras():
    with _cameras_lock:
        return jsonify([_camera_info(c) for c in _cameras.values()])


@app.route("/api/cameras", methods=["POST"])
def add_camera():
    if not _require_model():
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json(silent=True) or {}
    name   = str(data.get("name", "Camera")).strip()
    source = data.get("source", "").strip()

    if not source:
        return jsonify({"error": "source is required"}), 400

    # Allow integer device index (e.g. "0" for webcam)
    if source.isdigit():
        source = int(source)

    # Optional geolocation
    latitude = None
    longitude = None
    if "latitude" in data and data["latitude"] not in (None, ""):
        try:
            latitude = float(data["latitude"])
            if not (-90 <= latitude <= 90):
                return jsonify({"error": "latitude must be between -90 and 90"}), 400
        except (ValueError, TypeError):
            return jsonify({"error": "latitude must be a number"}), 400
    if "longitude" in data and data["longitude"] not in (None, ""):
        try:
            longitude = float(data["longitude"])
            if not (-180 <= longitude <= 180):
                return jsonify({"error": "longitude must be between -180 and 180"}), 400
        except (ValueError, TypeError):
            return jsonify({"error": "longitude must be a number"}), 400

    cam_id = str(uuid.uuid4())[:8]

    # Per-camera settings: merge global defaults with any overrides
    cam_overrides = {}
    if "settings" in data and isinstance(data["settings"], dict):
        for key, (typ, lo, hi) in _SETTINGS_SCHEMA.items():
            if key in data["settings"]:
                try:
                    val = typ(data["settings"][key])
                    if lo <= val <= hi:
                        cam_overrides[key] = val
                except (ValueError, TypeError):
                    pass

    with _settings_lock:
        s = {**_settings, **cam_overrides}

    if cam_overrides:
        _camera_settings[cam_id] = cam_overrides

    cam = CameraPipeline(
        camera_id=cam_id,
        name=name,
        source=source,
        model=_model,
        class_names=_class_names,
        device=_device,
        event_queue=_raw_event_queue,
        snapshot_dir=SNAPSHOT_DIR,
        threshold=s["threshold"],
        bg_history=s["bg_history"],
        bg_var_threshold=s["bg_var_threshold"],
        min_area=s["min_area"],
        max_distance=s["max_distance"],
        max_missed=s["max_missed"],
        stationary_distance=s["stationary_distance"],
        stationary_min_frames=s["stationary_min_frames"],
        jpeg_quality=s["jpeg_quality"],
        latitude=latitude,
        longitude=longitude,
    )
    cam.start()

    with _cameras_lock:
        _cameras[cam_id] = cam

    return jsonify(_camera_info(cam)), 201


@app.route("/api/cameras/<cam_id>", methods=["DELETE"])
def remove_camera(cam_id):
    with _cameras_lock:
        cam = _cameras.pop(cam_id, None)
    if cam is None:
        return jsonify({"error": "not found"}), 404
    cam.stop()
    _camera_settings.pop(cam_id, None)
    return jsonify({"deleted": cam_id})


@app.route("/api/cameras/<cam_id>", methods=["PATCH"])
def update_camera(cam_id):
    with _cameras_lock:
        cam = _cameras.get(cam_id)
    if cam is None:
        return jsonify({"error": "not found"}), 404

    data = request.get_json(silent=True) or {}

    # Update name
    if "name" in data:
        cam.name = str(data["name"]).strip() or cam.name

    # Update geolocation
    if "latitude" in data:
        if data["latitude"] is None or data["latitude"] == "":
            cam.latitude = None
        else:
            try:
                lat = float(data["latitude"])
                if -90 <= lat <= 90:
                    cam.latitude = lat
                else:
                    return jsonify({"error": "latitude must be between -90 and 90"}), 400
            except (ValueError, TypeError):
                return jsonify({"error": "latitude must be a number"}), 400

    if "longitude" in data:
        if data["longitude"] is None or data["longitude"] == "":
            cam.longitude = None
        else:
            try:
                lng = float(data["longitude"])
                if -180 <= lng <= 180:
                    cam.longitude = lng
                else:
                    return jsonify({"error": "longitude must be between -180 and 180"}), 400
            except (ValueError, TypeError):
                return jsonify({"error": "longitude must be a number"}), 400

    # Update per-camera settings (only hot-applicable: threshold, jpeg_quality)
    if "settings" in data and isinstance(data["settings"], dict):
        overrides = _camera_settings.get(cam_id, {})
        for key in ("threshold", "jpeg_quality"):
            if key in data["settings"]:
                typ, lo, hi = _SETTINGS_SCHEMA[key]
                try:
                    val = typ(data["settings"][key])
                    if lo <= val <= hi:
                        overrides[key] = val
                        if key == "threshold":
                            cam.threshold = val
                        elif key == "jpeg_quality":
                            cam.jpeg_quality = val
                except (ValueError, TypeError):
                    pass
        if overrides:
            _camera_settings[cam_id] = overrides

    return jsonify(_camera_info(cam))


@app.route("/api/cameras/<cam_id>/start", methods=["POST"])
def start_camera(cam_id):
    with _cameras_lock:
        cam = _cameras.get(cam_id)
    if cam is None:
        return jsonify({"error": "not found"}), 404
    cam.start()
    return jsonify(_camera_info(cam))


@app.route("/api/cameras/<cam_id>/stop", methods=["POST"])
def stop_camera(cam_id):
    with _cameras_lock:
        cam = _cameras.get(cam_id)
    if cam is None:
        return jsonify({"error": "not found"}), 404
    cam.stop()
    return jsonify(_camera_info(cam))


# ─────────────────────────────────────────────────────────────────────────────
# Route — MJPEG video stream
# ─────────────────────────────────────────────────────────────────────────────

# Module-level placeholder cache keyed by camera name
_placeholder_cache: Dict[str, bytes] = {}

def _mjpeg_generator(cam: CameraPipeline):
    """Yield multipart JPEG frames from the camera's frame queue."""
    if cam.name not in _placeholder_cache:
        _placeholder_cache[cam.name] = _make_placeholder(cam.name)
    placeholder = _placeholder_cache[cam.name]
    while True:
        try:
            jpeg = cam.frame_queue.get(timeout=0.5)
        except queue.Empty:
            # Send a placeholder if the pipeline hasn't started or is slow
            jpeg = placeholder
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        )


def _make_placeholder(name: str) -> bytes:
    """Generate a small grey JPEG with 'No signal' text."""
    import cv2, numpy as np
    img = (np.ones((240, 320, 3), dtype=np.uint8) * 30)
    cv2.putText(img, name, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 1)
    cv2.putText(img, "No signal", (80, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 1)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


@app.route("/video/<cam_id>")
def video_feed(cam_id):
    with _cameras_lock:
        cam = _cameras.get(cam_id)
    if cam is None:
        return "Camera not found", 404
    return Response(
        _mjpeg_generator(cam),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Route — Server-Sent Events
# ─────────────────────────────────────────────────────────────────────────────

def _sse_generator(client_id: str, client_queue: queue.Queue):
    try:
        # Send a heartbeat comment every 15 seconds to keep the connection alive
        while True:
            try:
                event = client_queue.get(timeout=15)
                data = json.dumps(event)
                yield f"data: {data}\n\n"
            except queue.Empty:
                yield ": heartbeat\n\n"
    finally:
        with _sse_lock:
            _sse_clients.pop(client_id, None)


@app.route("/events")
def sse_events():
    client_id = str(uuid.uuid4())
    client_queue: queue.Queue = queue.Queue(maxsize=50)
    with _sse_lock:
        _sse_clients[client_id] = client_queue
    return Response(
        _sse_generator(client_id, client_queue),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Route — Event log (REST)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/events")
def get_events():
    n = min(int(request.args.get("n", 100)), 500)
    # Serve from in-memory ring buffer instead of reading entire log file
    events = list(_recent_events)[:n]
    return jsonify(events)


# ─────────────────────────────────────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────────────────────────────────────

def _load_model_global(model_path: str, threshold: float) -> None:
    global _model, _class_names, _device
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if _device.type == "cpu":
        print("WARNING: No CUDA GPU detected — running on CPU. Inference will be slow.")
        print("         Consider using a machine with an NVIDIA GPU for real-time performance.")
    print(f"Device : {_device}")
    _model, _class_names = load_model(model_path, _device)

    # Pre-populate the ring buffer from existing log file
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        _recent_events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    with _settings_lock:
        _settings["threshold"] = threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CCTV Illegal Dumping Monitoring — Flask server")
    parser.add_argument("--model",     default="C:\\Users\\thath\\OneDrive - University of Bradford\\Industrial AI Project\\trash_v_6\\trash_v_6\\models\\best_model.pth", help="Path to MobileNetV3 checkpoint")
    parser.add_argument("--threshold", type=float, default=0.5,  help="Classifier trash confidence threshold")
    parser.add_argument("--host",      default="0.0.0.0",        help="Bind host")
    parser.add_argument("--port",      type=int,  default=5000,  help="Bind port")
    parser.add_argument("--debug",     action="store_true",       help="Flask debug mode")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _load_model_global(args.model, args.threshold)
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True, use_reloader=False)
