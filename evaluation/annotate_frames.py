"""
annotate_frames.py — Auto-annotate extracted frames using YOLOv8x (largest model).
Output: YOLO-format .txt files in data/annotations/raw_auto/

Usage:
    cd "Homework 2"
    python3 evaluation/annotate_frames.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

FRAMES_DIR = ROOT / "data" / "frames"
RAW_AUTO_DIR = ROOT / "data" / "annotations" / "raw_auto"


def annotate_frames(
    frames_dir: Path = FRAMES_DIR,
    output_dir: Path = RAW_AUTO_DIR,
    conf: float = 0.20,
) -> None:
    from ultralytics import YOLO

    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        print(f"No frames found in {frames_dir}")
        print("Run: python3 scripts/extract_frames.py")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Auto-annotating {len(frame_paths)} frames with YOLOv8x...")
    print(f"(This downloads yolov8x.pt on first run ~137MB)\n")

    model = YOLO("yolov8x.pt")

    total_boxes = 0
    for i, img_path in enumerate(frame_paths):
        results = model(str(img_path), conf=conf, verbose=False)
        r = results[0]

        txt_path = output_dir / (img_path.stem + ".txt")
        boxes_written = 0

        with open(txt_path, "w") as f:
            for box in r.boxes:
                cls = int(box.cls[0])
                cx, cy, w, h = box.xywhn[0].tolist()  # normalized coords
                conf_val = float(box.conf[0])
                f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                boxes_written += 1

        total_boxes += boxes_written

        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{len(frame_paths)}] {img_path.name} — {boxes_written} boxes")

    print(f"\nDone. Annotated {len(frame_paths)} frames, {total_boxes} total boxes.")
    print(f"Output: {output_dir}")
    print("\nNext step: python3 evaluation/review_annotations.py")


if __name__ == "__main__":
    annotate_frames()
