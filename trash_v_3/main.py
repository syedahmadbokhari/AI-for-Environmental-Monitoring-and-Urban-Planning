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


def run_pipeline(video_path: str, model_path: str, threshold: float) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    model, class_names = load_model(model_path, device)

    bg = BackgroundSubtractor()
    tracker = SimpleTracker()
    event_detector = EventDetector()

    frame_idx = 0
    confirmed_trash_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
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
            classify_roi(frame, track, model, class_names, device, threshold)
            if track.classifier_label == "trash":
                confirmed_trash_count += 1
                print(f"  [TRASH] frame {frame_idx}  track {track.track_id}  {track.classifier_confidence:.1%}")

        # Visualization
        vis = frame.copy()
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
    parser.add_argument("--video",     type=str,   default="C:\\Users\\thath\\OneDrive - University of Bradford\Industrial AI Project\\trash\\trash\\trash_v_3\\videos\\video.mp4", help="Path to input video.")
    parser.add_argument("--model",     type=str,   default="C:\\Users\\thath\OneDrive - University of Bradford\Industrial AI Project\\trash\\trash\\trash_v_3\\models\\best_model.pth",   help="Path to model checkpoint.")
    parser.add_argument("--threshold", type=float, default=0.5,               help="Classifier confidence threshold.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.video, args.model, args.threshold)
