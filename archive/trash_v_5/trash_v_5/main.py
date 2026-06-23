"""
main.py — CLI entry point for the standalone detection demo.
All pipeline logic lives in pipeline.py.

Usage:
    python main.py --video videos/video.mp4 --model best_model.pth --threshold 0.5
"""

import argparse

import cv2
import torch

from core.classifier import load_model
from core.pipeline import (
    BackgroundSubtractor,
    EventDetector,
    SimpleTracker,
    classify_roi,
    find_bounding_boxes,
)

# Limit PyTorch CPU threads to avoid starving the system
torch.set_num_threads(2)

PROCESS_EVERY_N = 3
PROCESS_WIDTH = 640


def run_pipeline(video_path: str, model_path: str, threshold: float) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: No CUDA GPU detected — running on CPU. Inference will be slow.")
    print(f"Device : {device}")
    model, class_names = load_model(model_path, device)

    bg = BackgroundSubtractor()
    tracker = SimpleTracker()
    event_detector = EventDetector()

    frame_idx = 0
    confirmed_trash_count = 0
    _scale = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Frame skip — only process every Nth frame
        if frame_idx % PROCESS_EVERY_N != 0:
            continue

        # Compute scale once
        if _scale is None:
            orig_w = frame.shape[1]
            if orig_w > PROCESS_WIDTH:
                _scale = PROCESS_WIDTH / orig_w
            else:
                _scale = 1.0

        # Downscale for detection
        if _scale < 1.0:
            small_frame = cv2.resize(frame, None, fx=_scale, fy=_scale,
                                     interpolation=cv2.INTER_AREA)
        else:
            small_frame = frame

        # Stage 1
        fg_mask = bg.apply(small_frame)
        # Stage 2
        bboxes = find_bounding_boxes(fg_mask, min_area=max(1, int(bg.min_area * _scale * _scale)))
        tracks = tracker.update(bboxes)
        # Stage 3
        new_events = event_detector.update(frame_idx, tracks)
        # Stage 4 — classify on full-res frame
        for _, track in new_events:
            if _scale < 1.0:
                ox, oy, ow, oh = track.bbox
                inv = 1.0 / _scale
                orig_bbox = track.bbox
                track.bbox = (int(ox * inv), int(oy * inv), int(ow * inv), int(oh * inv))
                classify_roi(frame, track, model, class_names, device, threshold)
                track.bbox = orig_bbox
            else:
                classify_roi(frame, track, model, class_names, device, threshold)
            if track.classifier_label == "trash":
                confirmed_trash_count += 1
                print(f"  [TRASH] frame {frame_idx}  track {track.track_id}  {track.classifier_confidence:.1%}")

        # Visualization — annotate in-place on the small frame
        vis = small_frame
        small_mask = cv2.resize(fg_mask, (0, 0), fx=0.25, fy=0.25)
        small_bgr = cv2.cvtColor(small_mask, cv2.COLOR_GRAY2BGR)
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
            elif t.stationary_frames >= tracker.stationary_min_frames:
                color, tag = (0, 255, 255), f"ID{t.track_id}"
            else:
                color, tag = (0, 255, 0), f"ID{t.track_id}"

            cv2.rectangle(vis, (x, y), (x + wb, y + hb), color, 2)
            cv2.putText(vis, tag, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        cv2.putText(
            vis,
            f"Frame: {frame_idx}  Drops: {len(event_detector.events)}  Trash: {confirmed_trash_count}",
            (mw + 6, 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
        )

        cv2.imshow("Trash Detection Pipeline", vis)
        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trash detection: standalone CLI demo.")
    parser.add_argument("--video",     type=str,   default="C:/Users/thath/OneDrive - University of Bradford/Industrial AI Project/trash_v_6", help="Path to input video.")
    parser.add_argument("--model",     type=str,   default="models/best_model.pth",   help="Path to model checkpoint.")
    parser.add_argument("--threshold", type=float, default=0.5,               help="Classifier confidence threshold.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.video, args.model, args.threshold)
