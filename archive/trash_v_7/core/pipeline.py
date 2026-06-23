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

import collections
import functools
import os
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
        self._last_stats: Optional[np.ndarray] = None  # cached per-frame component stats

    def apply(self, frame: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """Apply background subtraction and return a clean foreground mask.

        ``min_area`` is always interpreted as pixels on the **original
        resolution** frame.  The *scale* factor (original → processing
        resolution) is used to convert it to processing-frame pixels so
        that the UI value stays intuitive regardless of downscale ratio.
        """
        fg_mask = self.bg_subtractor.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # Use only 1 OPEN iteration so small blobs (far-away trash) survive
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Convert min_area from original-resolution pixels to downscaled pixels
        scaled_min_area = max(1, int(self.min_area * scale * scale))

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg_mask, 8)
        self._last_stats = stats  # cache all component stats before area filtering
        clean_mask = np.zeros_like(fg_mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= scaled_min_area:
                clean_mask[labels == i] = 255
        return clean_mask

    def get_small_blob_bboxes(
        self, scaled_frag_min: int, scaled_min: int
    ) -> List[Tuple[int, int, int, int]]:
        """Return bboxes of blobs with area in [scaled_frag_min, scaled_min).

        These blobs survived morphological filtering but were removed by the
        min_area threshold — they are candidate fragment detachments from
        tracked objects.  Both thresholds are in downscaled-frame pixels.
        """
        if self._last_stats is None or scaled_frag_min >= scaled_min:
            return []
        bboxes: List[Tuple[int, int, int, int]] = []
        for i in range(1, len(self._last_stats)):
            area = int(self._last_stats[i, cv2.CC_STAT_AREA])
            if scaled_frag_min <= area < scaled_min:
                x = int(self._last_stats[i, cv2.CC_STAT_LEFT])
                y = int(self._last_stats[i, cv2.CC_STAT_TOP])
                w = int(self._last_stats[i, cv2.CC_STAT_WIDTH])
                h = int(self._last_stats[i, cv2.CC_STAT_HEIGHT])
                bboxes.append((x, y, w, h))
        return bboxes


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
    # Refined trash blob bounding box (downscaled coords, x, y, w, h)
    trash_bbox: Optional[Tuple[int, int, int, int]] = None
    # Hierarchical blob grouping fields
    is_fragment: bool = False          # True when this is a detached small blob
    parent_group_id: Optional[int] = None  # track_id of the group it detached from
    # Number of raw blobs that were merged into this group track
    merged_blob_count: int = 1


class SimpleTracker:
    def __init__(
        self,
        max_distance: float = 50.0,
        max_missed: int = 15,
        stationary_distance: float = 2.0,
        stationary_min_frames: int = 30,
        merge_gap: float = 30.0,
        fragment_min_area: int = 100,
        max_event_area: int = 5000,
    ) -> None:
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.stationary_distance = stationary_distance
        self.stationary_min_frames = stationary_min_frames
        self.merge_gap = merge_gap
        self.fragment_min_area = fragment_min_area
        self.max_event_area = max_event_area
        # Dormant buffer: recently deleted tracks kept for re-identification
        self._dormant: Dict[int, Track] = {}
        self._dormant_ttl: Dict[int, int] = {}  # tid → remaining frames
        # Previous frame's per-group raw bboxes for detachment detection
        # Maps group_track_id → list of raw bboxes that composed the group
        self._prev_group_raw: Dict[int, List[Tuple[int, int, int, int]]] = {}

    @staticmethod
    def _centroid(x: int, y: int, w: int, h: int) -> Tuple[int, int]:
        return (int(x + w / 2), int(y + h / 2))

    @staticmethod
    def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        """Compute IoU between two (x, y, w, h) bounding boxes."""
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    def _match_cost(self, track: Track, bbox: Tuple[int, int, int, int],
                    centroid: Tuple[int, int]) -> float:
        """Combined cost: centroid distance penalised by lack of IoU overlap.

        Returns a cost in [0, inf).  Lower is better.
        A pure centroid matcher would ignore shape overlap and fail when a
        blob changes shape (person leaves, trash remains).  IoU captures
        spatial overlap so that even when the centroid jumps, high overlap
        keeps the match.
        """
        dist = float(np.linalg.norm(np.array(centroid) - np.array(track.centroid)))
        iou = self._iou(track.bbox, bbox)
        # If IoU > 0.3, halve the distance penalty so overlapping blobs match
        # even when the centroid shifts significantly.
        if iou > 0.3:
            dist *= 0.5
        # If IoU > 0 at all, cap distance at max_distance so it can still match
        if iou > 0.0:
            dist = min(dist, self.max_distance * 0.95)
        return dist

    def _hungarian_assign(
        self,
        track_ids: List[int],
        detections: List[Tuple[int, int, int, int]],
    ) -> List[Tuple[int, int]]:
        """Optimal bipartite matching using the Munkres/Hungarian algorithm.

        Returns list of (detection_index, track_id) pairs.
        Unmatched detections / tracks are not included.
        """
        n_det = len(detections)
        n_trk = len(track_ids)
        if n_det == 0 or n_trk == 0:
            return []

        cost = np.full((n_det, n_trk), 1e6, dtype=np.float64)
        for di, bbox in enumerate(detections):
            c = self._centroid(*bbox)
            for ti, tid in enumerate(track_ids):
                cost[di, ti] = self._match_cost(self.tracks[tid], bbox, c)

        # Simple Munkres: iterate over the smaller dimension greedily on
        # sorted costs.  For the small N typical here (<50 blobs) this is
        # effectively optimal and avoids a scipy dependency.
        matches: List[Tuple[int, int]] = []
        used_det = set()
        used_trk = set()
        # Flatten and sort all (cost, det_idx, trk_idx)
        flat = []
        for di in range(n_det):
            for ti in range(n_trk):
                flat.append((cost[di, ti], di, ti))
        flat.sort()
        for c_val, di, ti in flat:
            if di in used_det or ti in used_trk:
                continue
            if c_val >= self.max_distance:
                continue
            matches.append((di, track_ids[ti]))
            used_det.add(di)
            used_trk.add(ti)
        return matches

    def _try_reidentify(self, bbox: Tuple[int, int, int, int],
                        centroid: Tuple[int, int]) -> Optional[Track]:
        """Check dormant tracks for a spatial match; return reusable Track or None."""
        best_tid = None
        best_cost = self.max_distance * 1.5  # allow slightly larger radius
        for tid, dt in self._dormant.items():
            dist = float(np.linalg.norm(np.array(centroid) - np.array(dt.centroid)))
            iou = self._iou(dt.bbox, bbox)
            cost = dist * (0.5 if iou > 0.2 else 1.0)
            if cost < best_cost:
                best_cost = cost
                best_tid = tid
        if best_tid is not None:
            track = self._dormant.pop(best_tid)
            self._dormant_ttl.pop(best_tid, None)
            return track
        return None

    def update(
        self,
        detections: List[Tuple[int, int, int, int]],
        raw_bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
        groups: Optional[List[List[int]]] = None,
        fragment_extra_bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> Dict[int, Track]:
        """Track merged group bboxes and detect fragment detachments.

        Parameters
        ----------
        detections : list of (x, y, w, h)
            Merged group bounding boxes (output of ``merge_nearby_bboxes``).
        raw_bboxes : list of (x, y, w, h), optional
            The original un-merged bboxes.  Used together with *groups* for
            detachment detection.
        groups : list of list of int, optional
            Per-group indices into *raw_bboxes* that were merged into each
            entry of *detections*.
        fragment_extra_bboxes : list of (x, y, w, h), optional
            Blobs that were filtered out by min_area but are above
            fragment_min_area.  Spatially gated against previous group
            footprints to promote only genuine detachments to fragment tracks.
        """
        # ── Age dormant tracks and evict expired ones ─────────────────────
        for tid in list(self._dormant_ttl):
            self._dormant_ttl[tid] -= 1
            if self._dormant_ttl[tid] <= 0:
                self._dormant.pop(tid, None)
                del self._dormant_ttl[tid]

        # ── Track merged groups (same logic as before) ────────────────────
        if not self.tracks:
            for gi, bbox in enumerate(detections):
                x, y, w, h = bbox
                c = self._centroid(x, y, w, h)
                reused = self._try_reidentify(bbox, c)
                if reused is not None:
                    reused.centroid = c
                    reused.last_centroid = c
                    reused.bbox = bbox
                    reused.age += 1
                    reused.missed = 0
                    reused.merged_blob_count = len(groups[gi]) if groups else 1
                    self.tracks[reused.track_id] = reused
                else:
                    t = Track(
                        track_id=self.next_id, centroid=c, bbox=bbox, last_centroid=c,
                        merged_blob_count=len(groups[gi]) if groups else 1,
                    )
                    self.tracks[self.next_id] = t
                    self.next_id += 1
            self._update_group_raw(detections, raw_bboxes, groups)
            return self.tracks

        track_ids = list(self.tracks.keys())
        matches = self._hungarian_assign(track_ids, detections)
        updated = set()
        matched_det = set()

        for di, tid in matches:
            bbox = detections[di]
            x, y, w, h = bbox
            c = self._centroid(x, y, w, h)
            t = self.tracks[tid]
            move = float(np.linalg.norm(np.array(c) - np.array(t.centroid)))
            t.last_centroid = t.centroid
            t.centroid = c
            t.bbox = bbox
            t.age += 1
            t.missed = 0
            t.stationary_frames = t.stationary_frames + 1 if move < self.stationary_distance else 0
            t.merged_blob_count = len(groups[di]) if groups else 1
            updated.add(tid)
            matched_det.add(di)

        # ── Handle unmatched detections (split detection) ─────────────────
        unmatched_det_idxs = [i for i in range(len(detections)) if i not in matched_det]
        missed_tids = [tid for tid in track_ids if tid not in updated]

        for tid in list(missed_tids):
            t = self.tracks[tid]
            overlapping = []
            for di in unmatched_det_idxs:
                if self._iou(t.bbox, detections[di]) > 0.05:
                    overlapping.append(di)
            if len(overlapping) >= 1:
                best_di = None
                best_d = float('inf')
                for di in overlapping:
                    c = self._centroid(*detections[di])
                    d = float(np.linalg.norm(np.array(c) - np.array(t.centroid)))
                    if d < best_d:
                        best_d = d
                        best_di = di
                if best_di is not None and best_d < self.max_distance * 2:
                    bbox = detections[best_di]
                    c = self._centroid(*bbox)
                    move = float(np.linalg.norm(np.array(c) - np.array(t.centroid)))
                    t.last_centroid = t.centroid
                    t.centroid = c
                    t.bbox = bbox
                    t.age += 1
                    t.missed = 0
                    t.stationary_frames = t.stationary_frames + 1 if move < self.stationary_distance else 0
                    t.merged_blob_count = len(groups[best_di]) if groups else 1
                    updated.add(tid)
                    unmatched_det_idxs.remove(best_di)
                    matched_det.add(best_di)
                    missed_tids.remove(tid)

        # Remaining unmatched detections → try dormant re-id, then new tracks
        for di in unmatched_det_idxs:
            bbox = detections[di]
            x, y, w, h = bbox
            c = self._centroid(x, y, w, h)
            reused = self._try_reidentify(bbox, c)
            if reused is not None:
                reused.centroid = c
                reused.last_centroid = c
                reused.bbox = bbox
                reused.age += 1
                reused.missed = 0
                reused.merged_blob_count = len(groups[di]) if groups else 1
                self.tracks[reused.track_id] = reused
                updated.add(reused.track_id)
            else:
                _, _, bw, bh = bbox
                blob_area = bw * bh
                # Check if this new detection was left behind by a previous group
                # (person walked away, small object remained) → mark as fragment.
                is_frag = False
                parent_gid: Optional[int] = None
                if blob_area <= self.max_event_area and self._prev_group_raw:
                    for prev_tid, prev_raws in self._prev_group_raw.items():
                        for prev_raw in prev_raws:
                            if (self._iou(bbox, prev_raw) > 0.0
                                    or _bbox_gap(bbox, prev_raw) < self.merge_gap):
                                is_frag = True
                                parent_gid = prev_tid
                                break
                        if is_frag:
                            break
                t = Track(
                    track_id=self.next_id, centroid=c, bbox=bbox, last_centroid=c,
                    merged_blob_count=len(groups[di]) if groups else 1,
                    is_fragment=is_frag,
                    parent_group_id=parent_gid,
                )
                self.tracks[self.next_id] = t
                updated.add(self.next_id)
                self.next_id += 1

        # ── Secondary match: keep sub-threshold fragment tracks alive ─────
        # Fragments created from extra_bboxes (area < min_area) never appear
        # in merged_bboxes so the main Hungarian loop never matches them.
        # Match them here against fragment_extra_bboxes each frame so that
        # age and stationary_frames accumulate and EventDetector can fire.
        if fragment_extra_bboxes:
            _used_extra: set = set()
            for tid in list(missed_tids):
                t = self.tracks.get(tid)
                if t is None or not t.is_fragment:
                    continue
                best_bb = None
                best_d = self.max_distance
                for bb in fragment_extra_bboxes:
                    if bb in _used_extra:
                        continue
                    c_bb = self._centroid(*bb)
                    d = float(np.linalg.norm(np.array(c_bb) - np.array(t.centroid)))
                    if d < best_d:
                        best_d = d
                        best_bb = bb
                if best_bb is not None:
                    c = self._centroid(*best_bb)
                    move = float(np.linalg.norm(np.array(c) - np.array(t.centroid)))
                    t.last_centroid = t.centroid
                    t.centroid = c
                    t.bbox = best_bb
                    t.age += 1
                    t.missed = 0
                    t.stationary_frames = (
                        t.stationary_frames + 1 if move < self.stationary_distance else 0
                    )
                    updated.add(tid)
                    missed_tids.remove(tid)
                    _used_extra.add(best_bb)

        # ── Handle missed tracks ──────────────────────────────────────────
        for tid in missed_tids:
            t = self.tracks[tid]
            t.missed += 1
            if t.missed > self.max_missed:
                del self.tracks[tid]
                self._dormant[tid] = t
                self._dormant_ttl[tid] = self.max_missed * 2

        # Detect fragment detachments using PREVIOUS frame's group footprints
        # (must run before _update_group_raw overwrites _prev_group_raw)
        if raw_bboxes is not None:
            self._detect_fragments(
                detections, raw_bboxes, groups or [],
                fragment_extra_bboxes or [],
            )
        self._update_group_raw(detections, raw_bboxes, groups)
        return self.tracks

    def _update_group_raw(
        self,
        detections: List[Tuple[int, int, int, int]],
        raw_bboxes: Optional[List[Tuple[int, int, int, int]]],
        groups: Optional[List[List[int]]],
    ) -> None:
        """Store per-group raw bboxes for next frame's detachment check."""
        self._prev_group_raw.clear()
        if raw_bboxes is None or groups is None:
            return
        # Map each merged detection to the group-track that matched it
        for tid, t in self.tracks.items():
            if t.is_fragment:
                continue
            # Find which detection index this track is using
            for gi, bbox in enumerate(detections):
                if t.bbox == bbox:
                    self._prev_group_raw[tid] = [raw_bboxes[i] for i in groups[gi]]
                    break

    def _detect_fragments(
        self,
        detections: List[Tuple[int, int, int, int]],
        raw_bboxes: List[Tuple[int, int, int, int]],
        groups: List[List[int]],
        extra_bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> None:
        """Secondary fragment guard: catches detached blobs not tracked yet.

        Two sources of candidate blobs are processed:
        1. ``raw_bboxes`` — blobs at or above min_area that are unaccounted for
           by the main group tracker.
        2. ``extra_bboxes`` — blobs BELOW min_area but AT OR ABOVE
           fragment_min_area.  These are normally invisible to the rest of the
           pipeline; here they are spatially gated so only blobs that were
           recently part of (or adjacent to) a tracked group are promoted to
           fragment tracks, preventing background noise from leaking through.
        """
        if not self._prev_group_raw:
            return

        # Collect raw bbox indices already claimed by group tracks or existing fragments
        accounted_raw: set = set()
        for tid, t in self.tracks.items():
            if t.is_fragment:
                continue
            for gi, bbox in enumerate(detections):
                if t.bbox == bbox:
                    for idx in groups[gi]:
                        accounted_raw.add(idx)
                    break

        fragment_bboxes = {t.bbox for t in self.tracks.values() if t.is_fragment}

        for prev_tid, prev_raws in self._prev_group_raw.items():
            group_track = self.tracks.get(prev_tid) or self._dormant.get(prev_tid)
            if group_track is None:
                continue

            # Where the group WAS last frame (for was_inside)
            prev_group_union = functools.reduce(_union_bbox, prev_raws)
            # Where the group IS now (for now_separated)
            current_group_bbox = group_track.bbox

            for ri, raw_bb in enumerate(raw_bboxes):
                if ri in accounted_raw:
                    continue
                _, _, rw, rh = raw_bb
                raw_area = rw * rh
                if raw_area > self.max_event_area or raw_area < self.fragment_min_area:
                    continue
                if raw_bb in fragment_bboxes:
                    continue

                # Was this blob near/inside the previous group's coverage?
                was_inside = False
                for prev_raw in prev_raws:
                    if self._iou(raw_bb, prev_raw) > 0.0 or _bbox_gap(raw_bb, prev_raw) < self.merge_gap * 1.5:
                        was_inside = True
                        break
                if not was_inside:
                    if self._iou(raw_bb, prev_group_union) > 0.0 or _bbox_gap(raw_bb, prev_group_union) < self.merge_gap:
                        was_inside = True
                if not was_inside:
                    continue

                # Is it now spatially separated from the current (moved) group?
                now_separated = _bbox_gap(raw_bb, current_group_bbox) > self.merge_gap * 0.5
                if not now_separated:
                    continue

                c = self._centroid(*raw_bb)
                already_tracked = any(
                    t.is_fragment and self._iou(t.bbox, raw_bb) > 0.3
                    for t in self.tracks.values()
                )
                if already_tracked:
                    continue

                frag = Track(
                    track_id=self.next_id,
                    centroid=c,
                    bbox=raw_bb,
                    last_centroid=c,
                    is_fragment=True,
                    parent_group_id=prev_tid,
                    merged_blob_count=1,
                )
                self.tracks[self.next_id] = frag
                accounted_raw.add(ri)
                self.next_id += 1

        # ── Second pass: extra small-blob candidates (below min_area) ─────
        # These blobs are pre-filtered to [fragment_min_area, min_area) by the
        # caller.  Apply the same spatial-gate logic; use a bbox-tuple set to
        # track which ones have already been claimed (no integer index).
        if extra_bboxes:
            accounted_extra: set = set()
            for raw_bb in extra_bboxes:
                if raw_bb in accounted_extra:
                    continue
                _, _, rw, rh = raw_bb
                raw_area = rw * rh
                # Upper-area guard: should not exceed a full-person blob
                if raw_area > self.max_event_area:
                    continue
                if raw_bb in fragment_bboxes:
                    continue
                # Check if already tracked as a fragment (avoid duplicate)
                already_tracked = any(
                    t.is_fragment and self._iou(t.bbox, raw_bb) > 0.3
                    for t in self.tracks.values()
                )
                if already_tracked:
                    continue

                # Spatial gate: was this blob near a PREVIOUS group footprint?
                for prev_tid, prev_raws in self._prev_group_raw.items():
                    group_track = self.tracks.get(prev_tid) or self._dormant.get(prev_tid)
                    if group_track is None:
                        continue

                    prev_group_union = functools.reduce(_union_bbox, prev_raws)
                    current_group_bbox = group_track.bbox

                    was_inside = False
                    for prev_raw in prev_raws:
                        if (self._iou(raw_bb, prev_raw) > 0.0
                                or _bbox_gap(raw_bb, prev_raw) < self.merge_gap * 1.5):
                            was_inside = True
                            break
                    if not was_inside:
                        if (self._iou(raw_bb, prev_group_union) > 0.0
                                or _bbox_gap(raw_bb, prev_group_union) < self.merge_gap):
                            was_inside = True
                    if not was_inside:
                        continue

                    now_separated = _bbox_gap(raw_bb, current_group_bbox) > self.merge_gap * 0.5
                    if not now_separated:
                        continue

                    c = self._centroid(*raw_bb)
                    frag = Track(
                        track_id=self.next_id,
                        centroid=c,
                        bbox=raw_bb,
                        last_centroid=c,
                        is_fragment=True,
                        parent_group_id=prev_tid,
                        merged_blob_count=1,
                    )
                    self.tracks[self.next_id] = frag
                    accounted_extra.add(raw_bb)
                    self.next_id += 1
                    break  # blob claimed; move to next candidate


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Event Detection
# ─────────────────────────────────────────────────────────────────────────────

class EventDetector:
    def __init__(self, stationary_min_frames: int = 30, max_event_area: int = 5000,
                 maxlen: int = 500) -> None:
        self.stationary_min_frames = stationary_min_frames
        self.max_event_area = max_event_area
        self.events: collections.deque = collections.deque(maxlen=maxlen)

    def update(self, frame_idx: int, tracks: Dict[int, Track]) -> List[Tuple[int, Track]]:
        """Returns only events newly triggered this frame.

        Filtering rules:
        - Fragment tracks (detached from a group) always eligible.
        - Non-fragment tracks are only eligible if their area is below
          ``max_event_area`` (filters out person-sized blobs).
        - In both cases the track must be stationary long enough.
        """
        new_events: List[Tuple[int, Track]] = []
        for t in tracks.values():
            if t.is_stationary_event:
                continue
            if t.stationary_frames < self.stationary_min_frames or t.age < 2:
                continue
            # Area gate: large group blobs (persons) should not fire events
            _, _, tw, th = t.bbox
            blob_area = tw * th
            if not t.is_fragment and blob_area > self.max_event_area:
                continue
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


def _bbox_gap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Edge-to-edge gap between two (x, y, w, h) bboxes.  Returns 0 if overlapping."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    dx = max(0, max(ax1 - bx2, bx1 - ax2))
    dy = max(0, max(ay1 - by2, by1 - ay2))
    return float((dx ** 2 + dy ** 2) ** 0.5)


def _union_bbox(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """Return the smallest bbox enclosing both a and b."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    x1 = min(ax1, bx1)
    y1 = min(ay1, by1)
    x2 = max(ax1 + aw, bx1 + bw)
    y2 = max(ay1 + ah, by1 + bh)
    return (x1, y1, x2 - x1, y2 - y1)


def merge_nearby_bboxes(
    bboxes: List[Tuple[int, int, int, int]],
    merge_gap: float = 30.0,
) -> Tuple[List[Tuple[int, int, int, int]], List[List[int]]]:
    """Merge nearby bboxes into groups using union-find on edge gap distance.

    Returns:
        merged_bboxes: list of merged group bounding boxes
        groups: list of lists — each inner list has the indices into the
                original *bboxes* that were merged into that group
    """
    n = len(bboxes)
    if n == 0:
        return [], []

    # Union-Find
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if _bbox_gap(bboxes[i], bboxes[j]) <= merge_gap:
                union(i, j)

    # Collect groups
    group_map: Dict[int, List[int]] = {}
    for i in range(n):
        root = find(i)
        group_map.setdefault(root, []).append(i)

    merged_bboxes: List[Tuple[int, int, int, int]] = []
    groups: List[List[int]] = []
    for members in group_map.values():
        box = bboxes[members[0]]
        for idx in members[1:]:
            box = _union_bbox(box, bboxes[idx])
        merged_bboxes.append(box)
        groups.append(members)

    return merged_bboxes, groups


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — Classifier confirmation
# ─────────────────────────────────────────────────────────────────────────────

def classify_roi(
    frame: np.ndarray,
    track: Track,
    model,
    class_names,
    device,
    threshold: float,
    fg_mask: Optional[np.ndarray] = None,
    scale: float = 1.0,
) -> None:
    """Extract ROI and classify it.

    For **fragment** tracks the bbox *is* the trash blob, so we use it
    directly.  For non-fragment tracks we fall back to the original
    contour-search heuristic.
    """
    fh, fw = frame.shape[:2]
    tx, ty, tw, th = track.bbox  # downscaled coords
    inv = 1.0 / scale if scale > 0 else 1.0

    blob_bbox_ds = None  # (x, y, w, h) in downscaled coords

    if track.is_fragment:
        # Fragment track: its bbox already isolates the trash blob
        blob_bbox_ds = (tx, ty, tw, th)
    elif fg_mask is not None:
        # Non-fragment: search the mask for the contour closest to centroid
        mh, mw = fg_mask.shape[:2]
        mx1 = max(0, tx)
        my1 = max(0, ty)
        mx2 = min(mw, tx + tw)
        my2 = min(mh, ty + th)
        if mx2 > mx1 and my2 > my1:
            crop = fg_mask[my1:my2, mx1:mx2]
            contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                tcx, tcy = track.centroid
                best_cnt = None
                best_dist = float('inf')
                for cnt in contours:
                    if cv2.contourArea(cnt) <= 0:
                        continue
                    cx, cy, cw, ch = cv2.boundingRect(cnt)
                    ccx = mx1 + cx + cw / 2
                    ccy = my1 + cy + ch / 2
                    d = (ccx - tcx) ** 2 + (ccy - tcy) ** 2
                    if d < best_dist:
                        best_dist = d
                        best_cnt = cnt
                if best_cnt is not None:
                    bx, by, bw, bh = cv2.boundingRect(best_cnt)
                    blob_bbox_ds = (mx1 + bx, my1 + by, bw, bh)

    track.trash_bbox = blob_bbox_ds

    # ── Build the ROI for classification (full-resolution frame) ──────────
    if blob_bbox_ds is not None:
        rx, ry, rw, rh = blob_bbox_ds
    else:
        rx, ry, rw, rh = tx, ty, tw, th

    # Scale to full resolution
    rx, ry, rw, rh = int(rx * inv), int(ry * inv), int(rw * inv), int(rh * inv)

    # Add 20 % padding
    pad_x, pad_y = int(rw * 0.2), int(rh * 0.2)
    x1, y1 = max(0, rx - pad_x), max(0, ry - pad_y)
    x2, y2 = min(fw, rx + rw + pad_x), min(fh, ry + rh + pad_y)
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
        bg_history: int = 500,
        bg_var_threshold: int = 50,
        min_area: int = 500,
        max_distance: float = 50.0,
        max_missed: int = 15,
        stationary_distance: float = 2.0,
        stationary_min_frames: int = 30,
        jpeg_quality: int = 70,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        process_every_n: int = 3,
        process_width: int = 640,
        merge_gap: int = 30,
        fragment_min_area: int = 100,
        max_event_area: int = 5000,
        roi: Optional[Tuple[float, float, float, float]] = None,
    ) -> None:
        self.camera_id = camera_id
        self.name = name
        self.source = source
        self.model = model
        self.class_names = class_names
        self.device = device
        self.threshold = threshold
        self.bg_history = bg_history
        self.bg_var_threshold = bg_var_threshold
        self.min_area = min_area
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.stationary_distance = stationary_distance
        self.stationary_min_frames = stationary_min_frames
        self.jpeg_quality = jpeg_quality
        self.event_queue = event_queue
        self.snapshot_dir = snapshot_dir
        self.latitude = latitude
        self.longitude = longitude
        self.process_every_n = max(1, process_every_n)
        self.process_width = process_width
        self.merge_gap = merge_gap
        self.fragment_min_area = fragment_min_area
        self.max_event_area = max_event_area
        # roi: (rx1, ry1, rx2, ry2) as fractions [0.0–1.0] or None for full frame
        self.roi: Optional[Tuple[float, float, float, float]] = roi

        os.makedirs(self.snapshot_dir, exist_ok=True)

        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self.status: str = "stopped"   # "running" | "stopped" | "error"
        self.error_msg: str = ""
        self.confirmed_count: int = 0
        self.total_drops: int = 0
        self.debug_view: bool = False   # side-by-side mask view for tuning

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
        bg = BackgroundSubtractor(
            history=self.bg_history,
            var_threshold=self.bg_var_threshold,
            min_area=self.min_area,
        )
        tracker = SimpleTracker(
            max_distance=self.max_distance,
            max_missed=self.max_missed,
            stationary_distance=self.stationary_distance,
            stationary_min_frames=self.stationary_min_frames,
            merge_gap=self.merge_gap,
            fragment_min_area=self.fragment_min_area,
            max_event_area=self.max_event_area,
        )
        event_detector = EventDetector(
            stationary_min_frames=self.stationary_min_frames,
            max_event_area=self.max_event_area,
        )
        frame_idx = 0

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.status = "error"
            self.error_msg = f"Cannot open source: {self.source}"
            return

        # Reduce internal OpenCV read-ahead buffer
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # For video files, pace playback to the file's native FPS
        _is_file = isinstance(self.source, str) and not self.source.startswith("rtsp")
        _fps = cap.get(cv2.CAP_PROP_FPS) if _is_file else 0
        _frame_interval = 1.0 / _fps if _fps > 0 else 0

        # Pre-compute scale factor for processing resolution
        _scale = None  # computed on first frame
        _last_pw = self.process_width  # track changes for hot-sync

        try:
            while not self._stop_event.is_set():
                _t0 = time.monotonic()
                ret, frame = cap.read()
                if not ret:
                    # end of file — loop video, or mark error for streams
                    if isinstance(self.source, str) and not self.source.startswith("rtsp"):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        # Reset all detection state so the replayed video
                        # is processed fresh, as if the pipeline just started.
                        bg = BackgroundSubtractor(
                            history=self.bg_history,
                            var_threshold=self.bg_var_threshold,
                            min_area=self.min_area,
                        )
                        tracker = SimpleTracker(
                            max_distance=self.max_distance,
                            max_missed=self.max_missed,
                            stationary_distance=self.stationary_distance,
                            stationary_min_frames=self.stationary_min_frames,
                            merge_gap=self.merge_gap,
                            fragment_min_area=self.fragment_min_area,
                            max_event_area=self.max_event_area,
                        )
                        event_detector = EventDetector(
                            stationary_min_frames=self.stationary_min_frames,
                            max_event_area=self.max_event_area,
                        )
                        frame_idx = 0
                        _scale = None
                        continue
                    else:
                        self.status = "error"
                        self.error_msg = "Stream ended or disconnected"
                        break

                frame_idx += 1

                # Frame skip — read but don't process every frame
                if frame_idx % self.process_every_n != 0:
                    continue

                # Compute / recompute scale when process_width changes
                orig_w = frame.shape[1]
                _desired_pw = self.process_width
                if _scale is None or _desired_pw != _last_pw:
                    if orig_w > _desired_pw:
                        _scale = _desired_pw / orig_w
                    else:
                        _scale = 1.0
                    _last_pw = _desired_pw

                # ── Hot-sync settings from instance attrs (may be updated by Flask thread) ──
                bg.bg_subtractor.setHistory(self.bg_history)
                bg.bg_subtractor.setVarThreshold(self.bg_var_threshold)
                bg.min_area = self.min_area
                tracker.max_distance = self.max_distance
                tracker.max_missed = self.max_missed
                tracker.stationary_distance = self.stationary_distance
                tracker.stationary_min_frames = self.stationary_min_frames
                tracker.merge_gap = self.merge_gap
                tracker.fragment_min_area = self.fragment_min_area
                tracker.max_event_area = self.max_event_area
                event_detector.stationary_min_frames = self.stationary_min_frames
                event_detector.max_event_area = self.max_event_area

                # Downscale for detection pipeline (Stages 1-3)
                if _scale < 1.0:
                    small_frame = cv2.resize(frame, None, fx=_scale, fy=_scale,
                                             interpolation=cv2.INTER_AREA)
                else:
                    small_frame = frame

                # Stage 1
                fg_mask = bg.apply(small_frame, scale=_scale)

                # Apply ROI mask — zero out pixels outside the region of interest
                _roi = self.roi
                if _roi is not None:
                    _mh, _mw = fg_mask.shape[:2]
                    _rx1 = max(0, int(_roi[0] * _mw))
                    _ry1 = max(0, int(_roi[1] * _mh))
                    _rx2 = min(_mw, int(_roi[2] * _mw))
                    _ry2 = min(_mh, int(_roi[3] * _mh))
                    _roi_mask = np.zeros_like(fg_mask)
                    _roi_mask[_ry1:_ry2, _rx1:_rx2] = 255
                    fg_mask = cv2.bitwise_and(fg_mask, _roi_mask)

                # Stage 2 — find raw blobs, merge nearby, track groups, detect fragments
                _scaled_min = max(1, int(bg.min_area * _scale * _scale))
                raw_bboxes = find_bounding_boxes(fg_mask, min_area=_scaled_min)
                merged_bboxes, groups = merge_nearby_bboxes(raw_bboxes, merge_gap=tracker.merge_gap * _scale)
                # Collect sub-threshold blobs for fragment detachment detection.
                # These are blobs in [fragment_min_area, min_area) that were
                # erased by BackgroundSubtractor but may be detached body parts.
                _scaled_frag_min = max(1, int(self.fragment_min_area * _scale * _scale))
                frag_extra_bboxes = bg.get_small_blob_bboxes(_scaled_frag_min, _scaled_min)
                tracks = tracker.update(
                    merged_bboxes,
                    raw_bboxes=raw_bboxes,
                    groups=groups,
                    fragment_extra_bboxes=frag_extra_bboxes,
                )
                # Stage 3
                new_events = event_detector.update(frame_idx, tracks)
                # Stage 4 — use full-res frame for classifier ROI
                for _, track in new_events:
                    self.total_drops += 1
                    classify_roi(
                        frame, track, self.model, self.class_names,
                        self.device, self.threshold,
                        fg_mask=fg_mask, scale=_scale,
                    )
                    if track.classifier_label == "trash":
                        self.confirmed_count += 1
                        snapshot_file = self._save_snapshot(frame, track, frame_idx, _scale)
                        self._push_event(frame_idx, track, snapshot_file)

                # Annotate directly on the small frame (no copy)
                vis = self._annotate(small_frame, fg_mask, tracks, event_detector, frame_idx)

                # Encode and push to frame queue (drop oldest if full)
                ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                if ok:
                    jpeg = buf.tobytes()
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.frame_queue.put(jpeg)

                # Pace video files to their native FPS
                if _frame_interval > 0:
                    elapsed = time.monotonic() - _t0
                    wait = _frame_interval * self.process_every_n - elapsed
                    if wait > 0:
                        time.sleep(wait)

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
        vis = frame  # annotate in-place to avoid ~6 MB copy per frame
        fh, fw = vis.shape[:2]

        # Draw tracking overlays on vis
        for t in tracks.values():
            x, y, wb, hb = t.bbox
            if t.classifier_label == "trash":
                cv2.rectangle(vis, (x, y), (x + wb, y + hb), (120, 120, 120), 1)
                if t.trash_bbox is not None:
                    bx, by, bw, bh = t.trash_bbox
                    cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 0, 255), 3)
                    cv2.putText(vis, f"TRASH {t.classifier_confidence:.0%}",
                                (bx, by - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    cv2.rectangle(vis, (x, y), (x + wb, y + hb), (0, 0, 255), 3)
                    cv2.putText(vis, f"TRASH {t.classifier_confidence:.0%}",
                                (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255), 2, cv2.LINE_AA)
                continue
            elif t.classifier_label == "no_trash":
                color, tag = (255, 200, 0), f"ID{t.track_id} clear {t.classifier_confidence:.0%}"
            elif t.is_fragment and t.is_stationary_event:
                color, tag = (0, 165, 255), f"ID{t.track_id} FRAG stopped"
            elif t.is_fragment:
                color, tag = (0, 255, 255), f"ID{t.track_id} FRAG st:{t.stationary_frames}"
            elif t.is_stationary_event:
                color, tag = (0, 0, 255), f"ID{t.track_id} stopped"
            elif t.stationary_frames >= 30:
                color, tag = (0, 255, 255), f"ID{t.track_id} st:{t.stationary_frames}"
            else:
                color, tag = (0, 255, 0), f"ID{t.track_id}"

            cv2.rectangle(vis, (x, y), (x + wb, y + hb), color, 2)
            cv2.putText(vis, tag, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # Draw ROI rectangle only when debug view is active (configurer-only)
        if self.debug_view and self.roi is not None:
            _rx1 = max(0, int(self.roi[0] * fw))
            _ry1 = max(0, int(self.roi[1] * fh))
            _rx2 = min(fw - 1, int(self.roi[2] * fw))
            _ry2 = min(fh - 1, int(self.roi[3] * fh))
            cv2.rectangle(vis, (_rx1, _ry1), (_rx2, _ry2), (0, 255, 255), 2)
            cv2.putText(vis, "ROI", (_rx1 + 4, _ry1 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        if self.debug_view:
            # ── Debug: side-by-side (annotated | fg_mask with bboxes) ──────
            mask_bgr = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
            for t in tracks.values():
                x, y, wb, hb = t.bbox
                if t.is_fragment:
                    # Fill the fragment region with a distinct magenta tint so the
                    # separated blob is clearly visible against the white group blob.
                    clr = (255, 0, 255)  # magenta for fragments
                    label = f"F{t.track_id} st:{t.stationary_frames}"
                    overlay = mask_bgr.copy()
                    cv2.rectangle(overlay, (x, y), (x + wb, y + hb), clr, -1)
                    cv2.addWeighted(overlay, 0.4, mask_bgr, 0.6, 0, mask_bgr)
                    cv2.rectangle(mask_bgr, (x, y), (x + wb, y + hb), clr, 2)
                else:
                    clr = (0, 255, 0)    # green for groups
                    label = f"G{t.track_id} m:{t.merged_blob_count} st:{t.stationary_frames}"
                    cv2.rectangle(mask_bgr, (x, y), (x + wb, y + hb), clr, 2)
                cv2.putText(mask_bgr, label, (x, y - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, clr, 1, cv2.LINE_AA)
            # Labels on each panel
            cv2.putText(vis, "CAMERA VIEW", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(mask_bgr, "PROCESSING MASK", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            # HUD on mask panel
            hud = f"fr:{frame_idx} tracks:{len(tracks)} drops:{self.total_drops}"
            cv2.putText(mask_bgr, hud, (8, fh - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1, cv2.LINE_AA)
            # Stitch side by side
            vis = np.hstack((vis, mask_bgr))
        else:
            # Normal mode: small mask thumbnail in top-left corner
            small = cv2.resize(fg_mask, (0, 0), fx=0.25, fy=0.25)
            small_bgr = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
            mh, mw = small_bgr.shape[:2]
            vis[0:mh, 0:mw] = small_bgr

        # HUD overlay
        hud = f"{self.name} | fr:{frame_idx} drops:{self.total_drops} trash:{self.confirmed_count}"
        cv2.putText(vis, hud, (8, vis.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
        return vis

    def _save_snapshot(self, frame: np.ndarray, track: Track, frame_idx: int,
                        scale: float = 1.0) -> str:
        filename = f"{self.camera_id}_{frame_idx}_{track.track_id}.jpg"
        path = os.path.join(self.snapshot_dir, filename)

        # Draw a red box around the trash blob on a copy of the frame
        snap = frame.copy()
        fh, fw = snap.shape[:2]
        inv = 1.0 / scale if scale > 0 else 1.0
        if track.trash_bbox is not None:
            bx, by, bw, bh = track.trash_bbox
            bx, by, bw, bh = int(bx * inv), int(by * inv), int(bw * inv), int(bh * inv)
        else:
            bx, by, bw, bh = track.bbox
            bx, by, bw, bh = int(bx * inv), int(by * inv), int(bw * inv), int(bh * inv)
        # Clip to frame bounds
        x1, y1 = max(0, bx), max(0, by)
        x2, y2 = min(fw, bx + bw), min(fh, by + bh)
        cv2.rectangle(snap, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(snap, f"TRASH {track.classifier_confidence:.0%}",
                    (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imwrite(path, snap)
        return filename

    def _push_event(self, frame_idx: int, track: Track, snapshot_file: str) -> None:
        # Include trash blob bbox in full-resolution coords for the UI
        trash_box = None
        if track.trash_bbox is not None:
            bx, by, bw, bh = track.trash_bbox
            # trash_bbox is stored in downscaled coords; not scaling here
            # because _push_event doesn't know scale — store as-is
            trash_box = [bx, by, bw, bh]
        event = {
            "camera_id": self.camera_id,
            "camera_name": self.name,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "frame_idx": frame_idx,
            "track_id": track.track_id,
            "confidence": round(track.classifier_confidence, 4),
            "snapshot_url": f"/snapshots/{snapshot_file}",
            "latitude": self.latitude,
            "longitude": self.longitude,
            "trash_bbox": trash_box,
        }
        try:
            self.event_queue.put_nowait(event)
        except queue.Full:
            pass
