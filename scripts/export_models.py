"""
export_models.py
Run once to export YOLO11n and YOLOv8s to TorchScript and ONNX formats.
Models are saved to backend/models/.

Usage:
    cd "Homework 2"
    python3 scripts/export_models.py
"""

import sys
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "backend" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_IDS = ["yolo11n", "yolov8s"]


def export_model(model_id: str) -> None:
    from ultralytics import YOLO

    print(f"\n{'='*50}")
    print(f"Exporting {model_id}...")
    print(f"{'='*50}")

    model = YOLO(f"{model_id}.pt")

    # --- TorchScript ---
    print(f"  [1/2] Exporting TorchScript...")
    ts_result = model.export(
        format="torchscript",
        imgsz=640,
        optimize=False,  # optimize=True is for mobile CPU; False works best with MPS
    )
    ts_src = Path(ts_result)
    ts_dst = MODELS_DIR / f"{model_id}.torchscript"
    shutil.move(str(ts_src), str(ts_dst))
    print(f"         Saved → {ts_dst} ({ts_dst.stat().st_size / 1e6:.1f} MB)")

    # --- ONNX ---
    print(f"  [2/2] Exporting ONNX...")
    onnx_result = model.export(
        format="onnx",
        imgsz=640,
        simplify=True,
        dynamic=False,   # static shapes required for CoreML delegation
        opset=12,        # opset 12 has broadest CoreML op support
        nms=False,       # pre-NMS output [1, 84, 8400] for consistent postprocessing
    )
    onnx_src = Path(onnx_result)
    onnx_dst = MODELS_DIR / f"{model_id}.onnx"
    shutil.move(str(onnx_src), str(onnx_dst))
    print(f"         Saved → {onnx_dst} ({onnx_dst.stat().st_size / 1e6:.1f} MB)")

    # --- Keep .pt for PyTorch eager inference ---
    pt_src = Path(f"{model_id}.pt")
    pt_dst = MODELS_DIR / f"{model_id}.pt"
    if pt_src.exists() and not pt_dst.exists():
        shutil.move(str(pt_src), str(pt_dst))
        print(f"         Moved .pt → {pt_dst}")
    elif not pt_dst.exists():
        # Download directly to models dir
        model2 = YOLO(str(pt_dst))
        print(f"         .pt already at {pt_dst}")

    # --- Shape verification ---
    print(f"  Verifying output shapes...")
    _verify_shapes(model_id)


def _verify_shapes(model_id: str) -> None:
    import torch
    import numpy as np
    import onnxruntime as ort

    dummy = torch.zeros(1, 3, 640, 640)

    # TorchScript shape check
    ts_path = MODELS_DIR / f"{model_id}.torchscript"
    try:
        m = torch.jit.load(str(ts_path), map_location="cpu")
        m.eval()
        with torch.no_grad():
            out = m(dummy)
        if isinstance(out, (list, tuple)):
            out = out[0]
        print(f"         TorchScript output shape: {tuple(out.shape)}")
        assert out.shape[1] == 84, f"Expected 84 channels, got {out.shape[1]}"
        print(f"         TorchScript: OK")
    except Exception as e:
        print(f"         TorchScript verification FAILED: {e}", file=sys.stderr)

    # ONNX shape check
    onnx_path = MODELS_DIR / f"{model_id}.onnx"
    try:
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        out = sess.run(None, {input_name: dummy.numpy()})[0]
        print(f"         ONNX output shape: {tuple(out.shape)}")
        assert out.shape[1] == 84, f"Expected 84 channels, got {out.shape[1]}"
        print(f"         ONNX: OK")
    except Exception as e:
        print(f"         ONNX verification FAILED: {e}", file=sys.stderr)


def main() -> None:
    print("YOLO Model Export Script")
    print(f"Output directory: {MODELS_DIR}")

    for model_id in MODEL_IDS:
        export_model(model_id)

    print(f"\nAll models exported to {MODELS_DIR}")
    print("Files:")
    for f in sorted(MODELS_DIR.iterdir()):
        print(f"  {f.name:40s}  {f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
