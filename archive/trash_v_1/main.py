import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

from infer import load_model, predict


# =========================
# Stage 1 - Background Subtraction
# =========================

class BackgroundSubtractor:
    """
    Stage 1:
    - Maintains a background model
    - Produces a clean foreground mask for moving objects
    """

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
        # Raw foreground mask
        fg_mask = self.bg_subtractor.apply(frame)

        # Remove shadows (value 127 in MOG2) and small noise
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Morphological operations to clean noise and fill holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Remove very small blobs
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg_mask, 8)
        clean_mask = np.zeros_like(fg_mask)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= self.min_area:
                clean_mask[labels == i] = 255

        return clean_mask


# =========================
# Stage 2 - Simple Multi-Object Tracking
# =========================


@dataclass
class Track:
    track_id: int
    centroid: Tuple[int, int]
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    age: int = 0  # total frames seen
    missed: int = 0  # frames not matched
    stationary_frames: int = 0
    last_centroid: Tuple[int, int] = field(default_factory=lambda: (0, 0))
    is_stationary_event: bool = False
    # Stage 4 classifier result ("trash", "no_trash", or "" = not yet classified)
    classifier_label: str = ""
    classifier_confidence: float = 0.0


class SimpleTracker:
    """
    Stage 2:
    - Tracks blobs over time using centroid distance
    - Assigns stable IDs to each moving object
    """

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
    def _centroid_from_bbox(x: int, y: int, w: int, h: int) -> Tuple[int, int]:
        return (int(x + w / 2), int(y + h / 2))

    def update(self, detections: List[Tuple[int, int, int, int]]) -> Dict[int, Track]:
        """
        detections: list of bounding boxes (x, y, w, h)
        returns: dict mapping track_id -> Track
        """
        if not self.tracks:
            for bbox in detections:
                x, y, w, h = bbox
                c = self._centroid_from_bbox(x, y, w, h)
                self.tracks[self.next_id] = Track(
                    track_id=self.next_id,
                    centroid=c,
                    bbox=bbox,
                    last_centroid=c,
                )
                self.next_id += 1
            return self.tracks

        # Mark all tracks as not updated initially
        track_ids = list(self.tracks.keys())
        updated_tracks = set()

        # For each detection, find the closest existing track
        for bbox in detections:
            x, y, w, h = bbox
            c = self._centroid_from_bbox(x, y, w, h)

            best_id = None
            best_dist = self.max_distance

            for tid in track_ids:
                if tid in updated_tracks:
                    continue
                t = self.tracks[tid]
                dist = np.linalg.norm(np.array(c) - np.array(t.centroid))
                if dist < best_dist:
                    best_dist = dist
                    best_id = tid

            if best_id is not None:
                # Update existing track
                t = self.tracks[best_id]
                move_dist = np.linalg.norm(np.array(c) - np.array(t.centroid))

                t.last_centroid = t.centroid
                t.centroid = c
                t.bbox = bbox
                t.age += 1
                t.missed = 0

                # Stationary logic
                if move_dist < self.stationary_distance:
                    t.stationary_frames += 1
                else:
                    t.stationary_frames = 0

                updated_tracks.add(best_id)
            else:
                # Create new track
                self.tracks[self.next_id] = Track(
                    track_id=self.next_id,
                    centroid=c,
                    bbox=bbox,
                    last_centroid=c,
                )
                updated_tracks.add(self.next_id)
                self.next_id += 1

        # Increase missed counter for tracks not updated
        for tid, t in list(self.tracks.items()):
            if tid not in updated_tracks:
                t.missed += 1
                if t.missed > self.max_missed:
                    del self.tracks[tid]

        return self.tracks


# =========================
# Stage 3 - Basic Event Detection
# =========================

class EventDetector:
    """
    Stage 3:
    - Detects simple "abandoned blob" events:
      an object that becomes stationary for a while.
    - This is a simplified version of your trash-drop logic.
    """

    def __init__(self) -> None:
        self.events: List[Tuple[int, Track]] = []  # (frame_idx, track)

    def update(self, frame_idx: int, tracks: Dict[int, Track]) -> List[Tuple[int, Track]]:
        """Returns only the events newly triggered this frame."""
        new_events: List[Tuple[int, Track]] = []
        for t in tracks.values():
            if (
                not t.is_stationary_event
                and t.stationary_frames >= 4  # ~1 second at 30 FPS
                and t.age >= 2  # existed for some time before
            ):
                t.is_stationary_event = True
                self.events.append((frame_idx, t))
                new_events.append((frame_idx, t))
        return new_events


# =========================
# Utility: Contour to bounding boxes
# =========================

def find_bounding_boxes(mask: np.ndarray, min_area: int = 500) -> List[Tuple[int, int, int, int]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes: List[Tuple[int, int, int, int]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        bboxes.append((x, y, w, h))
    return bboxes


# =========================
# Stage 4 - Classifier confirmation
# =========================

def classify_roi(
    frame: np.ndarray,
    track: Track,
    model,
    class_names: List[str],
    device,
    threshold: float,
) -> None:
    """
    Crop the blob ROI (with padding) from frame and run the MobileNetV3
    classifier to confirm whether it is actually trash.
    Result is stored directly on the track object.
    """
    fh, fw = frame.shape[:2]
    x, y, w, h = track.bbox

    # 20% padding so the classifier has surrounding context
    pad_x = int(w * 0.2)
    pad_y = int(h * 0.2)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(fw, x + w + pad_x)
    y2 = min(fh, y + h + pad_y)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return

    label, confidence, _ = predict(roi, model, class_names, device, threshold)
    track.classifier_label = label
    track.classifier_confidence = confidence

    status = "TRASH CONFIRMED" if label == "trash" else "false alarm"
    print(f"  [Stage 4] Track {track.track_id}: {status}  ({confidence:.1%})")


# =========================
# Main pipeline loop
# =========================

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

        # Stage 1: foreground mask
        fg_mask = bg.apply(frame)

        # Stage 2: blob detection + tracking
        bboxes = find_bounding_boxes(fg_mask, min_area=bg.min_area)
        tracks = tracker.update(bboxes)

        # Stage 3: detect newly stationary blobs
        new_events = event_detector.update(frame_idx, tracks)

        # Stage 4: run classifier on each newly dropped blob
        for _, track in new_events:
            classify_roi(frame, track, model, class_names, device, threshold)
            if track.classifier_label == "trash":
                confirmed_trash_count += 1

        # Visualization
        vis = frame.copy()

        # Show foreground mask in a small corner
        small_mask = cv2.resize(fg_mask, (0, 0), fx=0.3, fy=0.3)
        small_mask_bgr = cv2.cvtColor(small_mask, cv2.COLOR_GRAY2BGR)
        mh, mw = small_mask_bgr.shape[:2]
        vis[0:mh, 0:mw] = small_mask_bgr

        for t in tracks.values():
            x, y, w_box, h_box = t.bbox

            if t.classifier_label == "trash":
                color = (255, 0, 255)   # Magenta  — confirmed trash
                tag = f"ID {t.track_id} TRASH {t.classifier_confidence:.0%}"
            elif t.classifier_label == "no_trash":
                color = (255, 200, 0)   # Cyan-blue — classifier rejected
                tag = f"ID {t.track_id} no_trash {t.classifier_confidence:.0%}"
            elif t.is_stationary_event:
                color = (0, 0, 255)     # Red       — stopped, classifier running
                tag = f"ID {t.track_id} stopped"
            elif t.stationary_frames >= tracker.stationary_min_frames:
                color = (0, 255, 255)   # Yellow    — stationary candidate
                tag = f"ID {t.track_id}"
            else:
                color = (0, 255, 0)     # Green     — moving
                tag = f"ID {t.track_id}"

            cv2.rectangle(vis, (x, y), (x + w_box, y + h_box), color, 2)
            cv2.putText(
                vis, tag, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
            )

        cv2.putText(
            vis,
            f"Frame: {frame_idx}  Drops: {len(event_detector.events)}  Trash: {confirmed_trash_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Trash Detection Pipeline", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trash detection: background subtraction + tracking + classifier.")
    parser.add_argument("--video",     type=str,   default="C:\\Users\\thath\\OneDrive - University of Bradford\\Industrial AI Project\\trash\\trash\\trash_v_1\\videos\\video.mp4",      help="Path to input video file.")
    parser.add_argument("--model",     type=str,   default="C:\\Users\\thath\\OneDrive - University of Bradford\\Industrial AI Project\\trash\\trash\\trash_v_1\\best_model.pth", help="Path to MobileNetV3 checkpoint.")
    parser.add_argument("--threshold", type=float, default=0.5,              help="Classifier trash confidence threshold (0-1).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.video, args.model, args.threshold)


