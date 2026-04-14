"""
main.py — FastAPI backend for object detection inference.

OpenAI-compatible API style:
  POST /v1/detect/image
  POST /v1/detect/video
  POST /v1/benchmark
  GET  /v1/models
  GET  /health

Run:
    cd backend
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import sys
import time
import uuid
import asyncio
import traceback
from contextlib import asynccontextmanager
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import cv2
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from backend.config import ModelName, Backend, MODELS_DIR, DEFAULT_CONF_THRESHOLD
from backend.detectors.base import BaseDetector
from backend.detectors.pytorch_detector import PyTorchDetector
from backend.detectors.torchscript_detector import TorchScriptDetector
from backend.detectors.onnx_detector import ONNXCoreMLDetector
from backend.utils.video_utils import (
    extract_frames_from_bytes,
    run_detection_on_frames,
    compute_latency_stats,
)

# ---------------------------------------------------------------------------
# Model registry — populated at startup
# ---------------------------------------------------------------------------
registry: dict[tuple[str, str], BaseDetector] = {}


def _make_detector(model_name: str, backend: str) -> BaseDetector:
    if backend == Backend.pytorch_cpu.value:
        return PyTorchDetector(model_name, device="cpu")
    elif backend == Backend.pytorch_mps.value:
        return PyTorchDetector(model_name, device="mps")
    elif backend == Backend.torchscript_cpu.value:
        return TorchScriptDetector(model_name)
    elif backend == Backend.onnx_coreml.value:
        return ONNXCoreMLDetector(model_name)
    else:
        raise ValueError(f"Unknown backend: {backend}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load all model-backend combinations at startup."""
    print("Loading models...")
    models_ready = MODELS_DIR.exists() and any(MODELS_DIR.glob("*.pt"))

    if not models_ready:
        print(
            "WARNING: backend/models/ is empty.\n"
            "Run: python3 scripts/export_models.py\n"
            "Attempting to load anyway (models will auto-download on first use)..."
        )

    load_errors: list[str] = []
    for model_name in ModelName:
        for backend in Backend:
            key = (model_name.value, backend.value)
            try:
                det = _make_detector(model_name.value, backend.value)
                # Load eagerly for pytorch backends; lazy-load TS and ONNX
                # to avoid blocking startup for too long
                if "pytorch" in backend.value:
                    det.load()
                registry[key] = det
                print(f"  Registered: {model_name.value} / {backend.value}")
            except Exception as e:
                load_errors.append(f"{model_name.value}/{backend.value}: {e}")
                print(f"  FAILED: {model_name.value} / {backend.value}: {e}")

    if load_errors:
        print(f"\n{len(load_errors)} backend(s) failed to load (will error on request).")
    print("Startup complete.\n")
    yield
    registry.clear()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Object Detection API",
    description="YOLO inference backend with TorchScript and ONNX/CoreML acceleration",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------
class DetectionItem(BaseModel):
    bbox: list[float] = Field(description="[x1, y1, x2, y2] in pixels")
    class_id: int
    class_name: str
    confidence: float


class ImageDetectionResponse(BaseModel):
    id: str
    model: str
    backend: str
    latency_ms: float
    image_width: int
    image_height: int
    detections: list[DetectionItem]


class FrameResult(BaseModel):
    frame_idx: int
    latency_ms: float
    detections: list[DetectionItem]


class VideoDetectionResponse(BaseModel):
    id: str
    model: str
    backend: str
    total_frames_processed: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    frames: list[FrameResult]


class BenchmarkRequest(BaseModel):
    model: ModelName = ModelName.yolo11n
    backend: Backend = Backend.pytorch_mps
    num_iterations: int = Field(default=100, ge=10, le=1000)
    image_size: int = Field(default=640, ge=320, le=1280)


class BenchmarkResponse(BaseModel):
    id: str
    model: str
    backend: str
    num_iterations: int
    mean_ms: float
    std_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    throughput_fps: float


class ModelInfo(BaseModel):
    id: str
    backends: list[str]
    loaded: bool


class ModelsResponse(BaseModel):
    models: list[ModelInfo]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_detector(model: str, backend: str) -> BaseDetector:
    key = (model, backend)
    if key not in registry:
        raise HTTPException(status_code=404, detail=f"Model '{model}' / backend '{backend}' not available")
    det = registry[key]
    # Lazy load for TS and ONNX backends
    if not det._loaded:
        try:
            det.load()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Failed to load model: {e}")
    return det


def _decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image file")
    return img


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    loaded = [f"{m}/{b}" for (m, b), det in registry.items() if det._loaded]
    return {
        "status": "ok",
        "mps_available": torch.backends.mps.is_available(),
        "loaded_models": loaded,
        "total_registered": len(registry),
    }


@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    model_map: dict[str, list[str]] = {}
    loaded_set: set[str] = set()
    for (m, b), det in registry.items():
        model_map.setdefault(m, []).append(b)
        if det._loaded:
            loaded_set.add(m)
    return ModelsResponse(
        models=[
            ModelInfo(id=m, backends=backends, loaded=m in loaded_set)
            for m, backends in model_map.items()
        ]
    )


@app.post("/v1/detect/image", response_model=ImageDetectionResponse)
async def detect_image(
    file: UploadFile = File(...),
    model: str = Form(default=ModelName.yolo11n.value),
    backend: str = Form(default=Backend.pytorch_mps.value),
    conf_threshold: float = Form(default=DEFAULT_CONF_THRESHOLD),
):
    data = await file.read()
    img = _decode_image(data)

    det = _get_detector(model, backend)

    # Same MPS/thread restriction applies to single image inference
    if "mps" in backend:
        result = det.predict(img, conf_threshold=conf_threshold)
    else:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: det.predict(img, conf_threshold=conf_threshold)
        )

    return ImageDetectionResponse(
        id=f"det_{uuid.uuid4().hex[:8]}",
        model=model,
        backend=backend,
        latency_ms=round(result.latency_ms, 3),
        image_width=result.image_width,
        image_height=result.image_height,
        detections=[
            DetectionItem(
                bbox=[round(c, 2) for c in d.bbox],
                class_id=d.class_id,
                class_name=d.class_name,
                confidence=round(d.confidence, 4),
            )
            for d in result.detections
        ],
    )


@app.post("/v1/detect/video", response_model=VideoDetectionResponse)
async def detect_video(
    file: UploadFile = File(...),
    model: str = Form(default=ModelName.yolo11n.value),
    backend: str = Form(default=Backend.pytorch_mps.value),
    conf_threshold: float = Form(default=DEFAULT_CONF_THRESHOLD),
    max_frames: int = Form(default=150),
):
    data = await file.read()
    det = _get_detector(model, backend)

    # MPS (Metal) operations do not work reliably from thread pool executors on macOS.
    # For MPS-backed detectors, run frame extraction + inference directly (blocking).
    # For CPU/ONNX backends, offload to an executor to keep the event loop free.
    uses_mps = "mps" in backend

    def _process():
        frames, _fps = extract_frames_from_bytes(data, max_frames=max_frames)
        frame_results = run_detection_on_frames(frames, det, conf_threshold=conf_threshold)
        return frame_results

    if uses_mps:
        frame_results = _process()
    else:
        loop = asyncio.get_running_loop()
        frame_results = await loop.run_in_executor(None, _process)
    stats = compute_latency_stats(frame_results)

    return VideoDetectionResponse(
        id=f"vid_{uuid.uuid4().hex[:8]}",
        model=model,
        backend=backend,
        total_frames_processed=len(frame_results),
        avg_latency_ms=stats.get("avg_latency_ms", 0),
        p50_latency_ms=stats.get("p50_latency_ms", 0),
        p95_latency_ms=stats.get("p95_latency_ms", 0),
        p99_latency_ms=stats.get("p99_latency_ms", 0),
        min_latency_ms=stats.get("min_latency_ms", 0),
        max_latency_ms=stats.get("max_latency_ms", 0),
        frames=[
            FrameResult(
                frame_idx=r["frame_idx"],
                latency_ms=r["latency_ms"],
                detections=[DetectionItem(**d) for d in r["detections"]],
            )
            for r in frame_results
        ],
    )


@app.post("/v1/benchmark", response_model=BenchmarkResponse)
async def benchmark(req: BenchmarkRequest):
    det = _get_detector(req.model.value, req.backend.value)

    dummy_bgr = np.zeros((640, 640, 3), dtype=np.uint8)

    def _run():
        latencies: list[float] = []
        for _ in range(req.num_iterations):
            t0 = time.perf_counter()
            det.predict(dummy_bgr)
            latencies.append((time.perf_counter() - t0) * 1000.0)
        return latencies

    loop = asyncio.get_running_loop()
    latencies = await loop.run_in_executor(None, _run)

    arr = np.array(latencies)
    mean = float(arr.mean())
    return BenchmarkResponse(
        id=f"bench_{uuid.uuid4().hex[:8]}",
        model=req.model.value,
        backend=req.backend.value,
        num_iterations=req.num_iterations,
        mean_ms=round(mean, 3),
        std_ms=round(float(arr.std()), 3),
        p50_ms=round(float(np.percentile(arr, 50)), 3),
        p95_ms=round(float(np.percentile(arr, 95)), 3),
        p99_ms=round(float(np.percentile(arr, 99)), 3),
        min_ms=round(float(arr.min()), 3),
        max_ms=round(float(arr.max()), 3),
        throughput_fps=round(1000.0 / mean, 2),
    )
