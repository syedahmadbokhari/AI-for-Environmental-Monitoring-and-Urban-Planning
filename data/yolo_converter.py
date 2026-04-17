"""
YOLO v8 → Binary Classification Dataset Converter
====================================================
Reads a YOLOv8 data.yaml and converts the dataset into binary classification:
  - trash/     → cropped bounding box regions (ALL 16 classes → "trash")
  - no_trash/  → background crops with ZERO overlap with any annotation

Usage:
    python yolo_to_classifier.py --yaml path/to/data.yaml --output path/to/output

Output structure:
    output/
        train/
            trash/
            no_trash/
        val/
            trash/
            no_trash/
        test/
            trash/
            no_trash/
"""

import os
import random
import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
MIN_CROP_SIZE = 64       # px — discard crops smaller than this
MAX_CROP_SIZE = 512      # px — clamp oversized crops
MAX_RETRIES   = 200      # rejection sampling attempts per no_trash crop
IOU_THRESHOLD = 0.0      # zero overlap tolerance


# ─────────────────────────────────────────────
# YAML loader
# ─────────────────────────────────────────────

def load_yaml(yaml_path):
    """
    Load data.yaml and resolve split image paths.

    Roboflow YOLOv8 packages use '../train/images' style paths, meaning
    data.yaml sits INSIDE the project folder and images are siblings:

        ProjectFolder/
            data.yaml
            train/images/
            valid/images/
            test/images/

    But the yaml paths say '../train/images' (relative to yaml location),
    which would incorrectly resolve one level above.

    This function tries multiple candidate paths in order:
      1. Exactly as written in yaml (relative to yaml dir)
      2. Relative to yaml dir but stripping leading '../'  ← Roboflow fix
      3. Just the folder name portion inside yaml dir

    Returns:
        splits : dict  { split_name: Path(images_dir) }
        nc     : int   number of original classes
        names  : list  original class names
    """
    yaml_path = Path(yaml_path).resolve()
    yaml_dir  = yaml_path.parent

    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    splits = {}
    for key in ["train", "val", "test"]:
        if key not in cfg:
            continue

        raw = cfg[key]  # e.g. "../train/images"

        # Build candidate paths to try in order
        candidates = [
            (yaml_dir / raw).resolve(),                        # as-is
            (yaml_dir / Path(raw).name).resolve(),             # just filename: "images"
            (yaml_dir / Path(*Path(raw).parts[-2:])).resolve(),# last 2 parts: "train/images"
            (yaml_dir / raw.lstrip("./")).resolve(),           # strip leading ../
        ]

        found = None
        for candidate in candidates:
            if candidate.is_dir():
                found = candidate
                break

        if found:
            splits[key] = found
            print(f"   Split '{key}' → {found}")
        else:
            print(f"   Split '{key}' not found. Tried:")
            for c in candidates:
                print(f"         {c}")

    return splits, cfg.get("nc", 0), cfg.get("names", [])


# ─────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────

def images_dir_to_labels_dir(images_dir):
    """
    Derive labels directory from images directory.
    YOLOv8 convention:  .../images/train  →  .../labels/train
    """
    parts = list(Path(images_dir).parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == "images":
            parts[i] = "labels"
            return Path(*parts)
    # Fallback: sibling labels/ directory
    p = Path(images_dir)
    return p.parent.parent / "labels" / p.name


# ─────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────

def yolo_to_pixel(box, img_w, img_h):
    """YOLO normalized [cx, cy, w, h] → pixel [x1, y1, x2, y2], clamped."""
    cx, cy, bw, bh = box
    x1 = int((cx - bw / 2) * img_w)
    y1 = int((cy - bh / 2) * img_h)
    x2 = int((cx + bw / 2) * img_w)
    y2 = int((cy + bh / 2) * img_h)
    return max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)


def compute_iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter  = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


def overlaps_any(candidate, boxes):
    return any(compute_iou(candidate, b) > IOU_THRESHOLD for b in boxes)


# ─────────────────────────────────────────────
# Label parsing  (class ID intentionally ignored → all classes = "trash")
# ─────────────────────────────────────────────

def parse_label_file(label_path):
    """Return list of [cx, cy, w, h] (normalized). Class ID ignored."""
    boxes = []
    p = Path(label_path)
    if not p.exists():
        return boxes
    with open(p, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                boxes.append(list(map(float, parts[1:5])))
    return boxes


# ─────────────────────────────────────────────
# Crop extraction
# ─────────────────────────────────────────────

def extract_trash_crops(image, pixel_boxes):
    """Crop each annotated box. Skip boxes smaller than MIN_CROP_SIZE."""
    crops = []
    for (x1, y1, x2, y2) in pixel_boxes:
        if (x2 - x1) < MIN_CROP_SIZE or (y2 - y1) < MIN_CROP_SIZE:
            continue
        crop = image[y1:y2, x1:x2]
        if crop.size > 0:
            crops.append(crop)
    return crops


def avg_crop_size(pixel_boxes):
    """Average annotation size, clamped to [MIN, MAX]."""
    widths  = [x2 - x1 for x1, y1, x2, y2 in pixel_boxes]
    heights = [y2 - y1 for x1, y1, x2, y2 in pixel_boxes]
    w = max(MIN_CROP_SIZE, min(MAX_CROP_SIZE, int(np.mean(widths))))
    h = max(MIN_CROP_SIZE, min(MAX_CROP_SIZE, int(np.mean(heights))))
    return w, h


def extract_no_trash_crops(image, pixel_boxes, n_crops, crop_size):
    """
    Rejection sampling: pick random positions, keep only those with
    zero overlap against all annotation boxes.
    """
    img_h, img_w = image.shape[:2]
    cw, ch = crop_size

    if img_w < cw or img_h < ch:
        return []

    crops, attempts = [], 0
    while len(crops) < n_crops and attempts < MAX_RETRIES:
        attempts += 1
        x1 = random.randint(0, img_w - cw)
        y1 = random.randint(0, img_h - ch)
        if overlaps_any([x1, y1, x1 + cw, y1 + ch], pixel_boxes):
            continue
        crop = image[y1:y1 + ch, x1:x1 + cw]
        if crop.size > 0:
            crops.append(crop)

    return crops


# ─────────────────────────────────────────────
# Save helper
# ─────────────────────────────────────────────

def save_crop(crop, out_dir, stem, idx):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(Path(out_dir) / f"{stem}_{idx:06d}.jpg"),
        crop,
        [cv2.IMWRITE_JPEG_QUALITY, 95]
    )


# ─────────────────────────────────────────────
# Per-split processing
# ─────────────────────────────────────────────

def process_split(images_dir, output_dir, split_name):
    images_dir = Path(images_dir)
    labels_dir = images_dir_to_labels_dir(images_dir)

    trash_dir    = Path(output_dir) / split_name / "trash"
    no_trash_dir = Path(output_dir) / split_name / "no_trash"

    image_exts  = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    image_files = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in image_exts)

    if not image_files:
        print(f"   No images found in {images_dir}")
        return 0, 0

    trash_count = no_trash_count = skipped = 0

    print(f"\n '{split_name}' — {len(image_files):,} images")
    print(f"   images : {images_dir}")
    print(f"   labels : {labels_dir}")

    for img_path in tqdm(image_files, desc=f"  {split_name}"):
        image = cv2.imread(str(img_path))
        if image is None:
            skipped += 1
            continue

        img_h, img_w = image.shape[:2]
        label_path   = labels_dir / img_path.with_suffix(".txt").name
        yolo_boxes   = parse_label_file(label_path)

        # Skip unannotated images — ambiguous ground truth
        if not yolo_boxes:
            skipped += 1
            continue

        pixel_boxes   = [yolo_to_pixel(b, img_w, img_h) for b in yolo_boxes]
        trash_crops   = extract_trash_crops(image, pixel_boxes)

        if not trash_crops:
            skipped += 1
            continue

        # no_trash crop size = average annotation size for THIS image
        no_trash_crops = extract_no_trash_crops(
            image, pixel_boxes,
            n_crops   = len(trash_crops),
            crop_size = avg_crop_size(pixel_boxes)
        )

        stem = img_path.stem
        for crop in trash_crops:
            save_crop(crop, trash_dir, stem, trash_count)
            trash_count += 1

        for crop in no_trash_crops:
            save_crop(crop, no_trash_dir, stem, no_trash_count)
            no_trash_count += 1

    # ── Split summary
    print(f"\n    '{split_name}' complete:")
    print(f"        trash    : {trash_count:,}")
    print(f"        no_trash : {no_trash_count:,}")
    print(f"        skipped  : {skipped:,}")

    minority_count = min(trash_count, no_trash_count)
    majority_count = max(trash_count, no_trash_count)
    if minority_count > 0:
        ratio    = majority_count / minority_count
        minority = "no_trash" if no_trash_count < trash_count else "trash"
        if ratio > 1.05:
            print(f"\n    Imbalance: {ratio:.2f}x  →  minority class = '{minority}'")
            print(f"      Recommended weighted loss weight for '{minority}': {ratio:.3f}")

    return trash_count, no_trash_count


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert YOLOv8 multi-class dataset → binary classification dataset"
    )
    parser.add_argument("--yaml",   required=True, help="Path to data.yaml")
    parser.add_argument("--output", required=True, help="Output root directory")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    splits, nc, names = load_yaml(args.yaml)

    print("=" * 60)
    print("  YOLO → Binary Classification Converter")
    print("=" * 60)
    print(f"  YAML     : {args.yaml}")
    print(f"  Output   : {args.output}")
    print(f"  Classes  : {nc}  →  all mapped to 'trash'")
    print(f"  Names    : {', '.join(names)}")
    print(f"  Splits   : {list(splits.keys())}")
    print(f"  Min crop : {MIN_CROP_SIZE}px  |  Max: {MAX_CROP_SIZE}px")
    print(f"  IoU tol  : {IOU_THRESHOLD} (zero overlap)")
    print("=" * 60)

    total_trash = total_no_trash = 0
    for split_name, images_dir in splits.items():
        t, nt = process_split(images_dir, args.output, split_name)
        total_trash    += t
        total_no_trash += nt

    print("\n" + "=" * 60)
    print("  GLOBAL SUMMARY")
    print("=" * 60)
    print(f"    Total trash    : {total_trash:,}")
    print(f"    Total no_trash : {total_no_trash:,}")
    print(f"    Dataset saved  : {args.output}")
    print("=" * 60)
    print("\n Done! Ready to train MobileNetV3.")


if __name__ == "__main__":
    main()
