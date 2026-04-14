"""
export_coco_json.py — Convert verified YOLO-format annotations to COCO JSON
and create the dataset.yaml for YOLO.val().

Output:
  data/annotations/coco_gt.json   — COCO format ground truth
  data/annotations/dataset.yaml   — for YOLO.val()
  data/labels/                    — YOLO .txt labels (symlinked structure)

Usage:
    cd "Homework 2"
    python3 evaluation/export_coco_json.py
"""

import sys
import json
import shutil
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

FRAMES_DIR = ROOT / "data" / "frames"
VERIFIED_DIR = ROOT / "data" / "annotations" / "verified"
OUTPUT_JSON = ROOT / "data" / "annotations" / "coco_gt.json"
DATASET_YAML = ROOT / "data" / "annotations" / "dataset.yaml"
LABELS_DIR = ROOT / "data" / "labels"

from backend.config import COCO_CLASS_NAMES


def yolo_to_coco_bbox(cx: float, cy: float, w: float, h: float,
                       img_w: int, img_h: int) -> list[float]:
    """Convert YOLO normalized xywh → COCO [x_min, y_min, width, height] in pixels."""
    x_min = (cx - w / 2) * img_w
    y_min = (cy - h / 2) * img_h
    return [round(x_min, 2), round(y_min, 2), round(w * img_w, 2), round(h * img_h, 2)]


def export_coco_json(
    frames_dir: Path = FRAMES_DIR,
    verified_dir: Path = VERIFIED_DIR,
    output_json: Path = OUTPUT_JSON,
    dataset_yaml_path: Path = DATASET_YAML,
    labels_dir: Path = LABELS_DIR,
) -> None:
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        print(f"No frames found in {frames_dir}")
        return

    verified_stems = {p.stem for p in verified_dir.glob("*.txt")}
    if not verified_stems:
        print(f"No verified annotations in {verified_dir}")
        print("Run: python3 evaluation/review_annotations.py")
        return

    labels_dir.mkdir(parents=True, exist_ok=True)

    # Build COCO-format structures
    coco_images: list[dict] = []
    coco_annotations: list[dict] = []
    ann_id = 1

    verified_frame_paths = [p for p in frame_paths if p.stem in verified_stems]
    print(f"Exporting {len(verified_frame_paths)} verified frames to COCO JSON...")

    for img_id, img_path in enumerate(verified_frame_paths, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        coco_images.append({
            "id": img_id,
            "file_name": img_path.name,
            "width": img_w,
            "height": img_h,
        })

        label_path = verified_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        with open(label_path) as f:
            lines = [l.strip() for l in f if l.strip()]

        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

            coco_bbox = yolo_to_coco_bbox(cx, cy, bw, bh, img_w, img_h)
            area = coco_bbox[2] * coco_bbox[3]

            coco_annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls,    # COCO category_id = COCO class index (0-indexed here)
                "bbox": coco_bbox,
                "area": round(area, 2),
                "iscrowd": 0,
            })
            ann_id += 1

        # Also copy .txt to labels/ for YOLO.val()
        shutil.copy(label_path, labels_dir / label_path.name)

    # COCO categories
    categories = [{"id": i, "name": name} for i, name in enumerate(COCO_CLASS_NAMES)]

    coco_dict = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(coco_dict, f, indent=2)
    print(f"COCO JSON → {output_json}")
    print(f"  Images     : {len(coco_images)}")
    print(f"  Annotations: {len(coco_annotations)}")

    # dataset.yaml for YOLO.val()
    # YOLO.val() expects:
    #   data/
    #     frames/        ← images (already exist)
    #     labels/        ← matching .txt files (just written above)
    dataset = {
        "path": str(ROOT / "data"),
        "val": "frames",
        "nc": 80,
        "names": COCO_CLASS_NAMES,
    }
    with open(dataset_yaml_path, "w") as f:
        yaml.dump(dataset, f, default_flow_style=False, sort_keys=False)
    print(f"dataset.yaml → {dataset_yaml_path}")
    print("\nNext step: python3 evaluation/evaluate_map.py")


if __name__ == "__main__":
    export_coco_json()
