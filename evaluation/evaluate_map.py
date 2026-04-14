"""
evaluate_map.py — Run mAP evaluation and latency benchmark across all model-backend combos.

Produces:
  data/results/evaluation_report.json — structured results
  data/results/evaluation_report.txt  — human-readable table

Usage:
    cd "Homework 2"
    python3 evaluation/evaluate_map.py
"""

import sys
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATASET_YAML = ROOT / "data" / "annotations" / "dataset.yaml"
RESULTS_DIR = ROOT / "data" / "results"

from backend.config import ModelName, Backend, MODELS_DIR
from backend.detectors.pytorch_detector import PyTorchDetector
from backend.detectors.torchscript_detector import TorchScriptDetector
from backend.detectors.onnx_detector import ONNXCoreMLDetector


@dataclass
class EvalResult:
    model: str
    backend: str
    map50: float
    map50_95: float
    precision: float
    recall: float
    mean_latency_ms: float
    p50_latency_ms: float
    p99_latency_ms: float
    throughput_fps: float


def load_detector(model_name: str, backend: str):
    if backend == Backend.pytorch_mps.value:
        det = PyTorchDetector(model_name, device="mps")
    elif backend == Backend.pytorch_cpu.value:
        det = PyTorchDetector(model_name, device="cpu")
    elif backend == Backend.torchscript_cpu.value:
        det = TorchScriptDetector(model_name)
    elif backend == Backend.onnx_coreml.value:
        det = ONNXCoreMLDetector(model_name)
    else:
        raise ValueError(f"Unknown backend: {backend}")
    det.load()
    return det


def run_latency_benchmark(
    detector,
    n_warmup: int = 20,
    n_iters: int = 100,
) -> dict:
    """Benchmark raw inference latency on a black 640x640 image."""
    import cv2
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)

    # Warmup
    for _ in range(n_warmup):
        detector.predict(dummy)

    latencies: list[float] = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        detector.predict(dummy)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(latencies)
    mean = float(arr.mean())
    return {
        "mean_latency_ms": round(mean, 3),
        "std_latency_ms": round(float(arr.std()), 3),
        "p50_latency_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_latency_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_latency_ms": round(float(np.percentile(arr, 99)), 3),
        "throughput_fps": round(1000.0 / mean, 2),
    }


def run_map_evaluation(model_name: str, device: str = "mps") -> dict:
    """
    Use YOLO.val() to compute mAP on the verified annotation set.
    Returns dict with mAP50, mAP50_95, precision, recall.
    """
    if not DATASET_YAML.exists():
        print(f"  WARNING: {DATASET_YAML} not found. Skipping mAP evaluation.")
        print("  Run: python3 evaluation/export_coco_json.py")
        return {"map50": -1, "map50_95": -1, "precision": -1, "recall": -1}

    from ultralytics import YOLO

    pt_path = MODELS_DIR / f"{model_name}.pt"
    model = YOLO(str(pt_path) if pt_path.exists() else f"{model_name}.pt")

    print(f"  Running YOLO.val() for {model_name} on {device}...")
    results = model.val(
        data=str(DATASET_YAML),
        split="val",
        conf=0.25,
        iou=0.5,
        device=device,
        verbose=False,
        plots=False,
        save=False,
        save_json=False,
    )

    return {
        "map50": round(float(results.box.map50), 4),
        "map50_95": round(float(results.box.map), 4),
        "precision": round(float(results.box.mp), 4),
        "recall": round(float(results.box.mr), 4),
    }


def print_table(results: list[EvalResult]) -> str:
    header = (
        f"{'Model':<12} {'Backend':<20} {'mAP50':>7} {'mAP50-95':>10} "
        f"{'P':>7} {'R':>7} {'Lat(ms)':>9} {'p50(ms)':>9} {'p99(ms)':>9} {'FPS':>7}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for r in results:
        map50 = f"{r.map50:.3f}" if r.map50 >= 0 else "  N/A "
        map50_95 = f"{r.map50_95:.3f}" if r.map50_95 >= 0 else "    N/A "
        prec = f"{r.precision:.3f}" if r.precision >= 0 else "  N/A "
        rec = f"{r.recall:.3f}" if r.recall >= 0 else "  N/A "
        lines.append(
            f"{r.model:<12} {r.backend:<20} {map50:>7} {map50_95:>10} "
            f"{prec:>7} {rec:>7} {r.mean_latency_ms:>9.1f} "
            f"{r.p50_latency_ms:>9.1f} {r.p99_latency_ms:>9.1f} {r.throughput_fps:>7.1f}"
        )
    lines.append(sep)
    return "\n".join(lines)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Configs: (model_name, backend_str)
    # mAP is evaluated per model (not per backend — same weights, same output)
    # Latency is per backend
    configs = [
        (ModelName.yolo11n.value, Backend.pytorch_mps.value),
        (ModelName.yolo11n.value, Backend.torchscript_cpu.value),
        (ModelName.yolo11n.value, Backend.onnx_coreml.value),
        (ModelName.yolov8s.value, Backend.pytorch_mps.value),
        (ModelName.yolov8s.value, Backend.torchscript_cpu.value),
        (ModelName.yolov8s.value, Backend.onnx_coreml.value),
    ]

    # Compute mAP once per model (shared weights)
    map_cache: dict[str, dict] = {}

    eval_results: list[EvalResult] = []

    for model_name, backend_str in configs:
        print(f"\n{'='*55}")
        print(f"  Model: {model_name}   Backend: {backend_str}")
        print(f"{'='*55}")

        # mAP (cached per model)
        if model_name not in map_cache:
            map_cache[model_name] = run_map_evaluation(model_name, device=device)
        map_scores = map_cache[model_name]

        # Load detector
        try:
            detector = load_detector(model_name, backend_str)
        except Exception as e:
            print(f"  ERROR loading detector: {e}")
            continue

        # Latency benchmark
        print(f"  Running latency benchmark (100 iterations)...")
        latency = run_latency_benchmark(detector, n_warmup=20, n_iters=100)
        print(f"  mean={latency['mean_latency_ms']:.1f}ms  "
              f"p50={latency['p50_latency_ms']:.1f}ms  "
              f"p99={latency['p99_latency_ms']:.1f}ms  "
              f"fps={latency['throughput_fps']:.1f}")

        eval_results.append(EvalResult(
            model=model_name,
            backend=backend_str,
            map50=map_scores["map50"],
            map50_95=map_scores["map50_95"],
            precision=map_scores["precision"],
            recall=map_scores["recall"],
            mean_latency_ms=latency["mean_latency_ms"],
            p50_latency_ms=latency["p50_latency_ms"],
            p99_latency_ms=latency["p99_latency_ms"],
            throughput_fps=latency["throughput_fps"],
        ))

    # Print and save results
    table_str = print_table(eval_results)
    print(f"\n\nEvaluation Results\n{table_str}\n")

    # Save JSON
    json_path = RESULTS_DIR / "evaluation_report.json"
    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in eval_results], f, indent=2)
    print(f"Saved JSON → {json_path}")

    # Save text table
    txt_path = RESULTS_DIR / "evaluation_report.txt"
    with open(txt_path, "w") as f:
        f.write("CMPE258 Homework 2 — Evaluation Report\n\n")
        f.write(table_str)
        f.write("\n\nNotes:\n")
        f.write("  - mAP computed using YOLO.val() on verified annotations (COCO 0.5:0.95)\n")
        f.write("  - Latency benchmarked on 640x640 black frame, 100 iterations after 20 warmup\n")
        f.write(f"  - Device: Apple M4, Backend MPS\n")
    print(f"Saved table → {txt_path}")


if __name__ == "__main__":
    main()
