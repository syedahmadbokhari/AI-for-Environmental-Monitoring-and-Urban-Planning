"""
tests/test_pipeline.py — Unit tests for core/pipeline.py (GitHub 4-stage version).

Pipeline stages under test:
  Stage 1  BackgroundSubtractor  (MOG2 wrapper)
  Stage 2  SimpleTracker         (centroid IoU tracker)
  Stage 3  EventDetector         (stationarity gate)
  Stage 4  classify_roi          (MobileNetV3 ROI classification)
  Wrapper  CameraPipeline        (constructor only — no thread started)
  Utility  find_bounding_boxes

Tests run in <5 s total; slow tests are marked @pytest.mark.slow.
"""
import queue
import time
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from core.pipeline import (
    BackgroundSubtractor,
    CameraPipeline,
    EventDetector,
    SimpleTracker,
    Track,
    classify_roi,
    find_bounding_boxes,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_track(
    track_id: int = 1,
    centroid: tuple = (100, 100),
    bbox: tuple = (80, 80, 40, 40),
    stationary_frames: int = 0,
    age: int = 0,
    is_stationary_event: bool = False,
) -> Track:
    t = Track(track_id=track_id, centroid=centroid, bbox=bbox)
    t.stationary_frames = stationary_frames
    t.age = age
    t.is_stationary_event = is_stationary_event
    return t


def _make_camera_pipeline(event_queue: queue.Queue, snapshot_dir: str) -> CameraPipeline:
    """CameraPipeline with a mocked model — thread is NOT started."""
    return CameraPipeline(
        camera_id="test01",
        name="TestCam",
        source="dummy.mp4",
        model=MagicMock(),
        class_names=["no_trash", "trash"],
        device=torch.device("cpu"),
        event_queue=event_queue,
        snapshot_dir=snapshot_dir,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CameraPipeline construction
# ─────────────────────────────────────────────────────────────────────────────

def test_pipeline_initializes(event_queue, tmp_dir):
    """CameraPipeline constructor completes without starting a thread or raising."""
    cam = _make_camera_pipeline(event_queue, str(tmp_dir))
    assert cam.status == "stopped"
    assert cam.camera_id == "test01"
    assert cam.confirmed_count == 0
    assert cam.total_drops == 0


def test_pipeline_default_settings(event_queue, tmp_dir):
    """Default pipeline parameters are within expected operating ranges."""
    cam = _make_camera_pipeline(event_queue, str(tmp_dir))
    assert 0.0 < cam.threshold <= 1.0
    assert cam.stationary_min_frames > 0
    assert cam.max_missed > 0
    assert cam.process_every_n >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — BackgroundSubtractor
# ─────────────────────────────────────────────────────────────────────────────

def test_pipeline_processes_frame():
    """BackgroundSubtractor.apply() returns a binary mask with the correct shape."""
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
    bg = BackgroundSubtractor()
    mask = bg.apply(frame)
    assert mask.shape == (240, 320)
    assert mask.dtype == np.uint8


def test_background_subtractor_mask_is_binary():
    """Mask pixels are either 0 (background) or 255 (foreground)."""
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    bg = BackgroundSubtractor()
    mask = bg.apply(frame)
    unique = set(np.unique(mask).tolist())
    assert unique.issubset({0, 255}), f"Unexpected pixel values: {unique}"


def test_empty_frame_handling():
    """find_bounding_boxes on a blank mask returns an empty list."""
    blank = np.zeros((240, 320), dtype=np.uint8)
    boxes = find_bounding_boxes(blank, min_area=500)
    assert boxes == []


def test_find_bounding_boxes_detects_blob():
    """A white rectangle on a black mask is found by find_bounding_boxes."""
    mask = np.zeros((240, 320), dtype=np.uint8)
    mask[50:100, 80:160] = 255  # 50×80 px = 4 000 px² area
    boxes = find_bounding_boxes(mask, min_area=100)
    assert len(boxes) >= 1
    x, y, w, h = boxes[0]
    assert w > 0 and h > 0


def test_min_area_filter():
    """Blobs smaller than min_area are rejected by find_bounding_boxes."""
    mask = np.zeros((240, 320), dtype=np.uint8)
    mask[50:55, 50:55] = 255  # 5×5 = 25 px² — below any reasonable min_area
    boxes = find_bounding_boxes(mask, min_area=500)
    assert boxes == []


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — SimpleTracker
# ─────────────────────────────────────────────────────────────────────────────

def test_tracker_update_empty():
    """Tracker with no detections returns an empty dict."""
    tracker = SimpleTracker()
    result = tracker.update([])
    assert result == {}


def test_tracker_creates_track():
    """A single detection creates one active track with the correct bbox."""
    tracker = SimpleTracker()
    detections = [(80, 80, 40, 40)]
    tracks = tracker.update(detections)
    assert len(tracks) == 1
    track = next(iter(tracks.values()))
    assert track.bbox == (80, 80, 40, 40)


def test_tracker_stationary_frames_increment():
    """A non-moving object accumulates stationary_frames across updates."""
    tracker = SimpleTracker(stationary_distance=100.0)
    bbox = [(100, 100, 30, 30)]
    for _ in range(5):
        tracks = tracker.update(bbox)
    track = next(iter(tracks.values()))
    # stationary_distance=100 means any movement < 100px counts as stationary
    assert track.stationary_frames >= 3


def test_tracker_drops_missing_tracks():
    """A track that goes unmatched for max_missed frames is removed."""
    tracker = SimpleTracker(max_missed=2)
    tracker.update([(50, 50, 20, 20)])   # create track
    for _ in range(4):                    # 4 frames with no matching detection
        tracks = tracker.update([])
    assert len(tracks) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — EventDetector
# ─────────────────────────────────────────────────────────────────────────────

def test_pipeline_detects_events():
    """EventDetector fires when stationary_frames >= 4 and age >= 2."""
    det = EventDetector()
    track = _make_track(stationary_frames=5, age=5)
    events = det.update(frame_idx=100, tracks={1: track})
    assert len(events) == 1
    assert events[0][1].is_stationary_event is True


def test_event_dict_structure():
    """Each event returned by EventDetector is a (frame_idx, Track) tuple."""
    det = EventDetector()
    track = _make_track(stationary_frames=5, age=5)
    events = det.update(frame_idx=42, tracks={1: track})
    assert len(events) == 1
    frame_idx, fired_track = events[0]
    assert frame_idx == 42
    assert isinstance(fired_track, Track)
    assert fired_track.track_id == 1


def test_event_fires_only_once():
    """EventDetector does not fire a second event for the same track."""
    det = EventDetector()
    track = _make_track(stationary_frames=10, age=10)
    tracks = {1: track}
    first = det.update(frame_idx=1, tracks=tracks)
    second = det.update(frame_idx=2, tracks=tracks)
    assert len(first) == 1
    assert len(second) == 0


def test_event_requires_minimum_stationary_frames():
    """Track with stationary_frames < 4 does not trigger an event."""
    det = EventDetector()
    track = _make_track(stationary_frames=3, age=10)
    events = det.update(frame_idx=10, tracks={1: track})
    assert events == []


def test_event_requires_minimum_age():
    """Track with age < 2 does not trigger an event."""
    det = EventDetector()
    track = _make_track(stationary_frames=10, age=1)
    events = det.update(frame_idx=10, tracks={1: track})
    assert events == []


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — classify_roi
# ─────────────────────────────────────────────────────────────────────────────

def test_pipeline_confidence_threshold(dummy_image_bgr):
    """classify_roi labels the track 'trash' or 'no_trash' based on threshold."""
    # Mock model returns logits heavily biased toward trash (softmax ≈ 0.993)
    mock_model = MagicMock()
    mock_model.return_value = torch.tensor([[0.0, 5.0]])

    track_high = _make_track(bbox=(80, 80, 64, 64))
    classify_roi(dummy_image_bgr, track_high, mock_model,
                 ["no_trash", "trash"], torch.device("cpu"), threshold=0.5)
    assert track_high.classifier_label == "trash"

    mock_model.return_value = torch.tensor([[0.0, 5.0]])
    track_low = _make_track(track_id=2, bbox=(80, 80, 64, 64))
    classify_roi(dummy_image_bgr, track_low, mock_model,
                 ["no_trash", "trash"], torch.device("cpu"), threshold=0.999)
    assert track_low.classifier_label == "no_trash"


def test_classify_roi_updates_track_in_place(dummy_image_bgr):
    """classify_roi sets classifier_label and classifier_confidence on the track."""
    mock_model = MagicMock()
    mock_model.return_value = torch.tensor([[0.0, 3.0]])
    track = _make_track(bbox=(50, 50, 80, 80))
    classify_roi(dummy_image_bgr, track, mock_model,
                 ["no_trash", "trash"], torch.device("cpu"), threshold=0.5)
    assert track.classifier_label in ("trash", "no_trash")
    assert 0.0 <= track.classifier_confidence <= 1.0


def test_classify_roi_skips_zero_area_bbox(dummy_image_bgr):
    """classify_roi with a degenerate zero-area bbox does not raise or crash."""
    mock_model = MagicMock()
    track = _make_track(bbox=(0, 0, 0, 0))  # zero-area ROI
    classify_roi(dummy_image_bgr, track, mock_model,
                 ["no_trash", "trash"], torch.device("cpu"), threshold=0.5)
    # Label stays empty — no prediction was made
    assert track.classifier_label == ""


# ─────────────────────────────────────────────────────────────────────────────
# CameraPipeline event dict
# ─────────────────────────────────────────────────────────────────────────────

def test_camera_pipeline_event_keys(event_queue, tmp_dir):
    """_push_event produces a dict with all required schema keys."""
    cam = _make_camera_pipeline(event_queue, str(tmp_dir))
    track = _make_track(track_id=7, bbox=(30, 30, 40, 40))
    track.classifier_label = "trash"
    track.classifier_confidence = 0.82
    cam._push_event(42, track, "snap.jpg")

    event = event_queue.get_nowait()
    for key in ("camera_id", "camera_name", "timestamp", "frame_idx",
                "track_id", "confidence", "snapshot_url"):
        assert key in event, f"Required key '{key}' missing from event dict"


# ─────────────────────────────────────────────────────────────────────────────
# Performance
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_frame_processing_speed():
    """BackgroundSubtractor + SimpleTracker together process a frame in < 100ms."""
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    bg = BackgroundSubtractor()
    tracker = SimpleTracker()

    # JIT warm-up
    for i in range(3):
        mask = bg.apply(frame)
        boxes = find_bounding_boxes(mask)
        tracker.update(boxes)

    t0 = time.monotonic()
    mask = bg.apply(frame)
    boxes = find_bounding_boxes(mask)
    tracker.update(boxes)
    elapsed_ms = (time.monotonic() - t0) * 1000

    assert elapsed_ms < 100, f"Frame took {elapsed_ms:.1f} ms (> 100 ms limit)"
