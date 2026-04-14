"""
video_utils.py — Video frame extraction and per-frame detection utilities.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np


def extract_frames_from_bytes(
    video_bytes: bytes,
    max_frames: int = 300,
) -> tuple[list[np.ndarray], float]:
    """
    Write video bytes to a temp file, extract up to max_frames frames
    (sampling evenly if the video is long), and return (frames, source_fps).
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        frames, fps = _read_frames(Path(tmp_path), max_frames)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return frames, fps


def _read_frames(video_path: Path, max_frames: int) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Determine sampling interval
    interval = max(1, total // max_frames)

    frames: list[np.ndarray] = []
    idx = 0

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            frames.append(frame)
        idx += 1

    cap.release()
    return frames, fps


def run_detection_on_frames(
    frames: list[np.ndarray],
    detector,
    conf_threshold: float = 0.25,
) -> list[dict]:
    """
    Run detector on each frame. Returns list of per-frame result dicts
    ready for JSON serialization.
    """
    results: list[dict] = []
    for i, frame in enumerate(frames):
        result = detector.predict(frame, conf_threshold=conf_threshold)
        results.append({
            "frame_idx": i,
            "latency_ms": round(result.latency_ms, 3),
            "detections": [
                {
                    "bbox": [round(c, 2) for c in d.bbox],
                    "class_id": d.class_id,
                    "class_name": d.class_name,
                    "confidence": round(d.confidence, 4),
                }
                for d in result.detections
            ],
        })
    return results


def compute_latency_stats(per_frame_results: list[dict]) -> dict[str, float]:
    latencies = [r["latency_ms"] for r in per_frame_results]
    if not latencies:
        return {}
    arr = np.array(latencies, dtype=np.float64)
    return {
        "avg_latency_ms": round(float(arr.mean()), 3),
        "p50_latency_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_latency_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_latency_ms": round(float(np.percentile(arr, 99)), 3),
        "min_latency_ms": round(float(arr.min()), 3),
        "max_latency_ms": round(float(arr.max()), 3),
    }
