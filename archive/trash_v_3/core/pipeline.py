"""
pipeline.py — Shared detection engine for the CCTV monitoring system.

Contains:
  - BackgroundSubtractor (Stage 1)
  - Track dataclass + SimpleTracker (Stage 2)
  - EventDetector (Stage 3)
  - classify_roi (Stage 4)
  - CameraPipeline — wraps the full loop in a daemon thread, feeds frames
    and events out via queues so Flask can consume them.
"""

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.classifier import load_model, predict


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Background Subtraction
# ─────────────────────────────────────────────────────────────────────────────

class BackgroundSubtractor:
    def __init__(
        self,
        history: int = 500,
        var_threshold: int = 50,
        detect_shadows: bool = True,
        min_area: int = 500,
    ) -> None:
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )
        self.min_area = min_area

    def apply(self, frame: np.ndarray) -> np.ndarray:
        fg_mask = self.bg_subtractor.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg_mask, 8)
        clean_mask = np.zeros_like(fg_mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= self.min_area:
                clean_mask[labels == i] = 255
        return clean_mask


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Tracking
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Track:
    track_id: int
    centroid: Tuple[int, int]
    bbox: Tuple[int, int, int, int]   # x, y, w, h
    age: int = 0
    missed: int = 0
    stationary_frames: int = 0
    last_centroid: Tuple[int, int] = field(default_factory=lambda: (0, 0))
    is_stationary_event: bool = False
    # Stage 4 result
    classifier_label: str = ""        # "trash" | "no_trash" | "" = not yet classified
    classifier_confidence: float = 0.0


class SimpleTracker:
    def __init__(
        self,
        max_distance: float = 50.0,
        max_missed: int = 15,
        stationary_distance: float = 2.0,
        stationary_min_frames: int = 30,
    ) -> None:
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.stationary_distance = stationary_distance
        self.stationary_min_frames = stationary_min_frames

    @staticmethod
    def _centroid(x: int, y: int, w: int, h: int) -> Tuple[int, int]:
        return (int(x + w / 2), int(y + h / 2))

    def update(self, detections: List[Tuple[int, int, int, int]]) -> Dict[int, Track]:
        if not self.tracks:
            for bbox in detections:
                x, y, w, h = bbox
                c = self._centroid(x, y, w, h)
                self.tracks[self.next_id] = Track(
                    track_id=self.next_id, centroid=c, bbox=bbox, last_centroid=c
                )
                self.next_id += 1
            return self.tracks

        track_ids = list(self.tracks.keys())
        updated = set()

        for bbox in detections:
            x, y, w, h = bbox
            c = self._centroid(x, y, w, h)
            best_id, best_dist = None, self.max_distance

            for tid in track_ids:
                if tid in updated:
                    continue
                dist = float(np.linalg.norm(np.array(c) - np.array(self.tracks[tid].centroid)))
                if dist < best_dist:
                    best_dist, best_id = dist, tid

            if best_id is not None:
                t = self.tracks[best_id]
                move = float(np.linalg.norm(np.array(c) - np.array(t.centroid)))
                t.last_centroid = t.centroid
                t.centroid = c
                t.bbox = bbox
                t.age += 1
                t.missed = 0
                t.stationary_frames = t.stationary_frames + 1 if move < self.stationary_distance else 0
                updated.add(best_id)
            else:
                c2 = self._centroid(x, y, w, h)
                self.tracks[self.next_id] = Track(
                    track_id=self.next_id, centroid=c2, bbox=bbox, last_centroid=c2
                )
                updated.add(self.next_id)
                self.next_id += 1

        for tid, t in list(self.tracks.items()):
            if tid not in updated:
                t.missed += 1
                if t.missed > self.max_missed:
                    del self.tracks[tid]

        return self.tracks


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Event Detection
# ─────────────────────────────────────────────────────────────────────────────

class EventDetector:
    def __init__(self) -> None:
        self.events: List[Tuple[int, Track]] = []

    def update(self, frame_idx: int, tracks: Dict[int, Track]) -> List[Tuple[int, Track]]:
        """Returns only events newly triggered this frame."""
        new_events: List[Tuple[int, Track]] = []
        for t in tracks.values():
            if (
                not t.is_stationary_event
                and t.stationary_frames >= 4
                and t.age >= 2
            ):
                t.is_stationary_event = True
                self.events.append((frame_idx, t))
                new_events.append((frame_idx, t))
        return new_events


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def find_bounding_boxes(mask: np.ndarray, min_area: int = 500) -> List[Tuple[int, int, int, int]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            bboxes.append(cv2.boundingRect(cnt))
    return bboxes


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — Classifier confirmation
# ─────────────────────────────────────────────────────────────────────────────

def classify_roi(frame: np.ndarray, track: Track, model, class_names, device, threshold: float) -> None:
    fh, fw = frame.shape[:2]
    x, y, w, h = track.bbox
    pad_x, pad_y = int(w * 0.2), int(h * 0.2)
    x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
    x2, y2 = min(fw, x + w + pad_x), min(fh, y + h + pad_y)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return
    label, confidence, _ = predict(roi, model, class_names, device, threshold)
    track.classifier_label = label
    track.classifier_confidence = confidence


# ─────────────────────────────────────────────────────────────────────────────
# CameraPipeline — wraps one camera source in a daemon thread
# ─────────────────────────────────────────────────────────────────────────────

class CameraPipeline:
    """
    Runs the full 4-stage detection pipeline for one camera source in a
    background thread.

    Outputs:
      frame_queue  — queue.Queue(maxsize=2) of JPEG bytes (annotated frame)
      event_queue  — shared queue passed in from CameraManager; pushes event
                     dicts when confirmed trash is detected
    """

    def __init__(
        self,
        camera_id: str,
        name: str,
        source,           # str path/URL or int device index
        model,
        class_names: List[str],
        device,
        event_queue: queue.Queue,
        snapshot_dir: str,
        threshold: float = 0.5,
    ) -> None:
        self.camera_id = camera_id
        self.name = name
        self.source = source
        self.model = model
        self.class_names = class_names
        self.device = device
        self.threshold = threshold
        self.event_queue = event_queue
        self.snapshot_dir = snapshot_dir

        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self.status: str = "stopped"   # "running" | "stopped" | "error"
        self.error_msg: str = ""
        self.confirmed_count: int = 0
        self.total_drops: int = 0

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ── public control ────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3)
        self.status = "stopped"
        # drain frame queue so consumers don't block
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

    # ── internal loop ─────────────────────────────────────────────────────────

    def _run(self) -> None:
        self.status = "running"
        bg = BackgroundSubtractor()
        tracker = SimpleTracker()
        event_detector = EventDetector()
        frame_idx = 0

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.status = "error"
            self.error_msg = f"Cannot open source: {self.source}"
            return

        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    # end of file — loop video, or mark error for streams
                    if isinstance(self.source, str) and not self.source.startswith("rtsp"):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        self.status = "error"
                        self.error_msg = "Stream ended or disconnected"
                        break

                frame_idx += 1

                # Stage 1
                fg_mask = bg.apply(frame)
                # Stage 2
                bboxes = find_bounding_boxes(fg_mask, min_area=bg.min_area)
                tracks = tracker.update(bboxes)
                # Stage 3
                new_events = event_detector.update(frame_idx, tracks)
                # Stage 4
                for _, track in new_events:
                    self.total_drops += 1
                    classify_roi(frame, track, self.model, self.class_names, self.device, self.threshold)
                    if track.classifier_label == "trash":
                        self.confirmed_count += 1
                        snapshot_file = self._save_snapshot(frame, track, frame_idx)
                        self._push_event(frame_idx, track, snapshot_file)

                # Annotate frame
                vis = self._annotate(frame, fg_mask, tracks, event_detector, frame_idx)

                # Encode and push to frame queue (drop oldest if full)
                ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    jpeg = buf.tobytes()
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.frame_queue.put(jpeg)

        finally:
            cap.release()
            if self.status == "running":
                self.status = "stopped"

    def _annotate(
        self,
        frame: np.ndarray,
        fg_mask: np.ndarray,
        tracks: Dict[int, Track],
        event_detector: EventDetector,
        frame_idx: int,
    ) -> np.ndarray:
        vis = frame.copy()

        # Foreground mask thumbnail (top-left corner)
        small = cv2.resize(fg_mask, (0, 0), fx=0.25, fy=0.25)
        small_bgr = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
        mh, mw = small_bgr.shape[:2]
        vis[0:mh, 0:mw] = small_bgr

        for t in tracks.values():
            x, y, wb, hb = t.bbox
            if t.classifier_label == "trash":
                color, tag = (255, 0, 255), f"ID{t.track_id} TRASH {t.classifier_confidence:.0%}"
            elif t.classifier_label == "no_trash":
                color, tag = (255, 200, 0), f"ID{t.track_id} clear {t.classifier_confidence:.0%}"
            elif t.is_stationary_event:
                color, tag = (0, 0, 255), f"ID{t.track_id} stopped"
            elif t.stationary_frames >= 30:
                color, tag = (0, 255, 255), f"ID{t.track_id}"
            else:
                color, tag = (0, 255, 0), f"ID{t.track_id}"

            cv2.rectangle(vis, (x, y), (x + wb, y + hb), color, 2)
            cv2.putText(vis, tag, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # HUD overlay
        hud = f"{self.name} | fr:{frame_idx} drops:{self.total_drops} trash:{self.confirmed_count}"
        cv2.putText(vis, hud, (mw + 6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
        return vis

    def _save_snapshot(self, frame: np.ndarray, track: Track, frame_idx: int) -> str:
        import os
        os.makedirs(self.snapshot_dir, exist_ok=True)
        filename = f"{self.camera_id}_{frame_idx}_{track.track_id}.jpg"
        path = os.path.join(self.snapshot_dir, filename)
        cv2.imwrite(path, frame)
        return filename

    def _push_event(self, frame_idx: int, track: Track, snapshot_file: str) -> None:
        event = {
            "camera_id": self.camera_id,
            "camera_name": self.name,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "frame_idx": frame_idx,
            "track_id": track.track_id,
            "confidence": round(track.classifier_confidence, 4),
            "snapshot_url": f"/snapshots/{snapshot_file}",
        }
        try:
            self.event_queue.put_nowait(event)
        except queue.Full:
            pass
