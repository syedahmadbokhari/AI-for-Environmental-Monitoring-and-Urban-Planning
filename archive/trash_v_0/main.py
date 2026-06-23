import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np


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
        for t in tracks.values():
            if (
                not t.is_stationary_event
                and t.stationary_frames > 0
                and t.stationary_frames >= 4  # ~1 second at 30 FPS
                and t.age >= 2  # existed for some time before
            ):
                t.is_stationary_event = True
                self.events.append((frame_idx, t))
        return self.events


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
# Main demo loop
# =========================

def run_pipeline(video_path: str) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return

    bg = BackgroundSubtractor()
    tracker = SimpleTracker()
    event_detector = EventDetector()

    frame_idx = 0

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

        # Stage 3: event detection
        events = event_detector.update(frame_idx, tracks)

        # Visualization
        vis = frame.copy()

        # Show foreground mask in a small corner
        small_mask = cv2.resize(fg_mask, (0, 0), fx=0.3, fy=0.3)
        small_mask_bgr = cv2.cvtColor(small_mask, cv2.COLOR_GRAY2BGR)
        h, w = small_mask_bgr.shape[:2]
        vis[0:h, 0:w] = small_mask_bgr

        for t in tracks.values():
            x, y, w_box, h_box = t.bbox
            color = (0, 255, 0)  # moving object
            if t.stationary_frames >= tracker.stationary_min_frames:
                color = (0, 255, 255)  # stationary candidate
            if t.is_stationary_event:
                color = (0, 0, 255)  # confirmed event

            cv2.rectangle(vis, (x, y), (x + w_box, y + h_box), color, 2)
            label = f"ID {t.track_id}"
            cv2.putText(
                vis,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            vis,
            f"Frame: {frame_idx}  Events: {len(events)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Stages 1-2-3 Demo", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stages 1-3: background, tracking, and basic event detection.")
    parser.add_argument(
        "--video",
        type=str,
        default="C:\\Users\\thath\\OneDrive - University of Bradford\\Industrial AI Project\\trash\\trash\\trash_v_0\\videos\\2.mp4",
        help="Path to input video file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.video)


