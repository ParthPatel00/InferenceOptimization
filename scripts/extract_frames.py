"""
extract_frames.py
Extract frames from source_video.mp4 at 1 fps and save as JPGs.

Usage:
    cd "Homework 2"
    python3 scripts/extract_frames.py [--fps 1] [--max 200]
"""

import argparse
import cv2
from pathlib import Path


ROOT = Path(__file__).parent.parent
VIDEO_PATH = ROOT / "data" / "videos" / "source_video.mp4"
FRAMES_DIR = ROOT / "data" / "frames"


def extract_frames(
    video_path: Path,
    output_dir: Path,
    target_fps: float = 1.0,
    max_frames: int = 200,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = total_frames / source_fps if source_fps > 0 else 0

    print(f"Video: {video_path.name}")
    print(f"  Resolution : {width}x{height}")
    print(f"  Source FPS : {source_fps:.2f}")
    print(f"  Duration   : {duration_s:.1f}s ({total_frames} frames)")
    print(f"  Target FPS : {target_fps}")

    # How many source frames to skip between saves
    frame_interval = max(1, round(source_fps / target_fps))
    expected_count = min(max_frames, total_frames // frame_interval)
    print(f"  Frame interval : every {frame_interval} frames")
    print(f"  Expected output: ~{expected_count} frames")
    print()

    saved_paths: list[Path] = []
    frame_idx = 0
    saved = 0

    while saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            out_path = output_dir / f"frame_{saved:06d}.jpg"
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_paths.append(out_path)
            saved += 1
            if saved % 20 == 0 or saved == 1:
                print(f"  Saved {saved}/{expected_count}: {out_path.name}")

        frame_idx += 1

    cap.release()
    print(f"\nDone. Extracted {saved} frames to {output_dir}")
    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frames from video")
    parser.add_argument("--video", type=str, default=str(VIDEO_PATH))
    parser.add_argument("--output", type=str, default=str(FRAMES_DIR))
    parser.add_argument("--fps", type=float, default=1.0, help="Target frames per second to extract")
    parser.add_argument("--max", type=int, default=200, help="Maximum number of frames to extract")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: Video not found at {video_path}")
        print("Please download a video and place it at:")
        print(f"  {VIDEO_PATH}")
        return

    extract_frames(
        video_path=video_path,
        output_dir=Path(args.output),
        target_fps=args.fps,
        max_frames=args.max,
    )


if __name__ == "__main__":
    main()
