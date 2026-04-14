"""
review_annotations.py — Interactive OpenCV tool to review and verify auto-annotations.

Controls:
  a / Enter  — Accept frame annotations as-is
  d          — Delete ALL annotations for this frame (mark as background)
  n          — Skip to next frame without saving changes
  b          — Go back to previous frame
  q          — Quit and save progress

Accepted frames are written to data/annotations/verified/

Usage:
    cd "Homework 2"
    python3 evaluation/review_annotations.py
"""

import sys
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

FRAMES_DIR = ROOT / "data" / "frames"
RAW_AUTO_DIR = ROOT / "data" / "annotations" / "raw_auto"
VERIFIED_DIR = ROOT / "data" / "annotations" / "verified"

from backend.config import COCO_CLASSES

# Color palette: one distinct color per class (deterministic HSV)
def class_color(class_id: int) -> tuple[int, int, int]:
    hue = int((class_id * 47) % 180)
    rgb = cv2.cvtColor(
        np.array([[[hue, 220, 200]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
    )[0][0]
    return (int(rgb[0]), int(rgb[1]), int(rgb[2]))


def draw_boxes(img: np.ndarray, label_path: Path) -> np.ndarray:
    overlay = img.copy()
    h, w = img.shape[:2]

    if not label_path.exists():
        return overlay

    with open(label_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        color = class_color(cls)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        label = COCO_CLASSES.get(cls, f"cls{cls}")
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(overlay, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            overlay, label, (x1 + 2, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

    return overlay


def review_annotations(
    frames_dir: Path = FRAMES_DIR,
    raw_dir: Path = RAW_AUTO_DIR,
    verified_dir: Path = VERIFIED_DIR,
) -> None:
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        print(f"No frames found in {frames_dir}")
        return

    verified_dir.mkdir(parents=True, exist_ok=True)

    # Load progress — skip already-verified frames
    progress_file = verified_dir / ".progress.json"
    reviewed: set[str] = set()
    if progress_file.exists():
        with open(progress_file) as f:
            reviewed = set(json.load(f).get("reviewed", []))
        print(f"Resuming: {len(reviewed)} frames already reviewed.")

    print(f"\nReviewing {len(frame_paths)} frames.")
    print("Controls: [a/Enter]=Accept  [d]=Delete annotations  [n]=Skip  [b]=Back  [q]=Quit\n")

    idx = 0
    while idx < len(frame_paths):
        img_path = frame_paths[idx]
        label_path = raw_dir / (img_path.stem + ".txt")

        img = cv2.imread(str(img_path))
        if img is None:
            idx += 1
            continue

        # Resize for display if too large
        max_display = 900
        h, w = img.shape[:2]
        scale = min(1.0, max_display / max(h, w))
        display = cv2.resize(img, (int(w * scale), int(h * scale)))

        display_with_boxes = draw_boxes(display, label_path)

        # Count boxes
        n_boxes = 0
        if label_path.exists():
            with open(label_path) as f:
                n_boxes = sum(1 for l in f if l.strip())

        status_text = "VERIFIED" if img_path.stem in reviewed else "pending"
        cv2.putText(
            display_with_boxes,
            f"[{idx+1}/{len(frame_paths)}] {img_path.name}  boxes={n_boxes}  ({status_text})",
            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2
        )
        cv2.putText(
            display_with_boxes,
            "a=Accept  d=Delete  n=Skip  b=Back  q=Quit",
            (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1
        )

        cv2.imshow("Annotation Review", display_with_boxes)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("a"), 13):  # Accept
            # Copy label to verified (even if empty)
            dst_label = verified_dir / (img_path.stem + ".txt")
            if label_path.exists():
                shutil.copy(label_path, dst_label)
            else:
                dst_label.write_text("")
            reviewed.add(img_path.stem)
            idx += 1

        elif key == ord("d"):  # Delete annotations
            dst_label = verified_dir / (img_path.stem + ".txt")
            dst_label.write_text("")  # empty = no detections
            reviewed.add(img_path.stem)
            idx += 1

        elif key == ord("n"):  # Skip
            idx += 1

        elif key == ord("b"):  # Back
            idx = max(0, idx - 1)

        elif key == ord("q"):  # Quit
            break

        # Save progress after each action
        with open(progress_file, "w") as f:
            json.dump({"reviewed": list(reviewed)}, f)

    cv2.destroyAllWindows()
    print(f"\nReview complete. {len(reviewed)} frames verified.")
    print(f"Output: {verified_dir}")
    print("\nNext step: python3 evaluation/export_coco_json.py")


if __name__ == "__main__":
    review_annotations()
