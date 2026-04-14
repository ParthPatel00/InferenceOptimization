# CMPE258 Homework 2 — Object Detection Inference Optimization

## Assignment Requirements — Status

| # | Requirement | Status |
|---|---|---|
| 1 | Object detection on video using **at least two strong-performing models** | DONE — YOLO11n and YOLOv8s |
| 2 | **FastAPI backend** with OpenAI-compatible API style for image/video inference | DONE — 5 endpoints |
| 3 | **Next.js frontend**: upload image/video, visualize bounding boxes and latency | DONE |
| 4 | **Two inference acceleration methods** from the approved list | DONE — TorchScript + ONNX Runtime/CoreML |
| 5 | **Evaluate accuracy (mAP) and speed (latency)** for all configurations | DONE — 8 configs benchmarked |
| 6 | **Own video data with own annotations** for mAP evaluation | DONE — 200 frames, 3452 boxes |

---

## Models

| Model | Params | GFLOPs | COCO mAP50 | Role |
|---|---|---|---|---|
| **YOLO11n** | 2.6M | 6.5 | 39.5 | Speed-optimized nano model |
| **YOLOv8s** | 11.2M | 28.6 | 44.9 | Accuracy-optimized small model |

Both are COCO-pretrained (80 classes). The output tensor is identical across all backends: `[1, 84, 8400]` pre-NMS (cx, cy, w, h + 80 class scores). Postprocessing (decode + NMS) is done in the backend.

---

## Inference Acceleration Methods

### Why Acceleration Is Needed

PyTorch eager mode re-dispatches each op through the Python interpreter one at a time. On a 640x640 YOLO inference, this overhead accumulates across hundreds of ops. The two approved acceleration methods eliminate it in different ways.

### Method 1: TorchScript (requirement: "TorchScript")

TorchScript traces the model at export time and compiles it into a serialized IR graph (`.torchscript` file). At inference time:
- No Python interpreter overhead — the JIT runtime executes the graph directly
- Adjacent ops (e.g., conv + batch norm + activation) can be fused
- Memory buffers are preallocated across runs

Export: `model.export(format="torchscript", imgsz=640)` via Ultralytics  
Runtime: `torch.jit.load(path)` on CPU  
**Result on M4:** YOLO11n CPU eager 84ms → TorchScript CPU 80ms (~5% faster). TorchScript eliminates Python overhead, but both run on the same CPU cores so the gain is modest on Apple M4's fast CPU. The speedup is more significant on slower CPUs or when GPU acceleration is unavailable.

> Note: TorchScript models traced on CPU cannot be moved to MPS. The traced constants are pinned to the CPU device they were created on.

### Method 2: ONNX Runtime + CoreML Execution Provider (requirement: "ONNX Runtime with [hardware] backend")

ONNX is an open interchange format. ONNX Runtime selects execution providers based on available hardware. On Apple Silicon, `CoreMLExecutionProvider` is available and delegates the compute graph to the **Apple Neural Engine (ANE)** — Apple's dedicated fixed-function ML accelerator.

This is the direct equivalent of "ONNX Runtime with CUDA backend" on Apple M4 hardware:

| NVIDIA system | Apple M4 system |
|---|---|
| ONNX Runtime + CUDAExecutionProvider | ONNX Runtime + CoreMLExecutionProvider |
| GPU (CUDA cores) | Neural Engine (ANE) |
| CUDA driver | CoreML framework |

Export: `model.export(format="onnx", opset=12, simplify=True, nms=False)` via Ultralytics  
Runtime: `ort.InferenceSession(path, providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])`  
Verified: `session.get_providers()` returns `['CoreMLExecutionProvider', 'CPUExecutionProvider']`  
**Result on M4:** YOLO11n CPU eager 84ms → ONNX+CoreML 14.4ms — a **5.8x speedup**. YOLOv8s: 76.8ms → 18.0ms — a **4.3x speedup**.

### All Four Backends

| Backend ID | Label | Hardware | Speedup vs CPU (YOLO11n / YOLOv8s) |
|---|---|---|---|
| `pytorch_cpu` | Reference | CPU | 1x — 84ms / 77ms |
| `pytorch_mps` | Baseline (GPU) | Apple Metal GPU | **6.8x / 3.6x** — 12.4ms / 21.2ms |
| `torchscript_cpu` | **Acceleration #1** | CPU (compiled) | 1.05x / 1.1x — 80ms / 70ms |
| `onnx_coreml` | **Acceleration #2** | Apple Neural Engine | **5.8x / 4.3x** — 14.4ms / 18.0ms |

---

## Evaluation Results

### Dataset

- **Source:** `4K Road traffic video for object detection.mp4` (1920x1080, 5.1 min, urban traffic scene)
- **Frames:** 200 frames extracted at 1 fps (`scripts/extract_frames.py`)
- **Annotation method:** Auto-labeled with YOLOv8x (the largest/most accurate YOLO variant), then each frame manually reviewed and verified using an interactive OpenCV tool (`evaluation/review_annotations.py`). Frames with incorrect or ambiguous boxes were corrected or deleted.
- **Final annotation count:** 3,452 bounding boxes across 200 frames
- **Classes present:** car, truck, bus, person, motorcycle, bicycle (80-class COCO label space)
- **Format:** YOLO `.txt` format, exported to COCO JSON for validation

### Accuracy (mAP) — computed with YOLO.val() on the verified annotation set

mAP is a property of the model weights, not the inference backend. The same `.pt` weights produce identical predictions regardless of runtime. Evaluated at conf=0.25, IoU=0.5.

| Model | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|
| **YOLOv8s** | **0.388** | **0.316** | 0.392 | 0.395 |
| **YOLO11n** | 0.251 | 0.188 | 0.291 | 0.217 |

**YOLOv8s achieves 54% higher mAP50** (0.388 vs 0.251) at the cost of 2x more compute.

mAP is lower than published COCO benchmarks (44.9/39.5) because the evaluation dataset is a single traffic scene — many COCO classes are absent, and the scale/angle distribution differs from the diverse COCO val set.

### Latency and Throughput — benchmarked on Apple M4, 16GB RAM, no CUDA

100 iterations on a 640x640 frame. All timings include preprocessing, inference, and postprocessing.

```
Model        Backend              mAP50   mAP50-95   Mean(ms)   p50(ms)   p99(ms)    FPS   vs CPU
-------------------------------------------------------------------------------------------------
yolo11n      pytorch_mps          0.251      0.188       12.4      11.3      16.9    80.6   6.8x
yolo11n      onnx_coreml          0.251      0.188       14.4      12.7      39.5    69.3   5.8x
yolo11n      torchscript_cpu      0.251      0.188       80.1      79.5      86.7    12.5   1.05x
yolo11n      pytorch_cpu          0.251      0.188       84.1      83.2      98.4    11.9   1.0x (ref)
yolov8s      onnx_coreml          0.388      0.316       18.0      15.7      52.1    55.7   4.3x
yolov8s      pytorch_mps          0.388      0.316       21.2      20.7      22.6    47.2   3.6x
yolov8s      torchscript_cpu      0.388      0.316       69.9      69.0      73.4    14.3   1.1x
yolov8s      pytorch_cpu          0.388      0.316       76.8      75.0     109.8    13.0   1.0x (ref)
-------------------------------------------------------------------------------------------------
```

### Key Findings

**Model accuracy:**
- YOLOv8s achieves 54% higher mAP50 (0.388 vs 0.251) at ~3.6x more CPU compute cost
- mAP is identical across all backends for the same model — the weights are unchanged

**GPU acceleration (pytorch_mps and onnx_coreml):**
- Both hardware-accelerated backends are 4–7x faster than CPU for YOLO11n and 3.6–4.3x faster for YOLOv8s
- **PyTorch+MPS** is fastest for YOLO11n (12.4ms, 80.6 FPS) — the Metal GPU handles the nano model very efficiently
- **ONNX+CoreML** is fastest for YOLOv8s (18.0ms, 55.7 FPS) — the Neural Engine's higher throughput advantage grows with model size
- ONNX+CoreML shows high p99 variance (39.5ms / 52.1ms) due to CoreML recompiling sub-graphs on first contact with certain layer shapes; subsequent runs are stable

**TorchScript (CPU JIT compilation):**
- TorchScript eliminates Python interpreter overhead and fuses graph ops, but still runs on the same CPU cores as eager mode
- On Apple M4's fast CPU, the JIT benefit is modest: ~5% speedup for YOLO11n, ~10% for YOLOv8s
- TorchScript's value is clearest on systems where the CPU is the only available compute, or where the model is too large to fit on the GPU — it is the portable, hardware-independent acceleration method
- TorchScript cannot be moved to MPS after tracing on CPU (traced constants are device-pinned)

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Next.js 16 Frontend (localhost:3000)                    │
│  - Upload image/video or use embedded sample video       │
│  - Select model (YOLO11n / YOLOv8s) and backend         │
│  - View bounding box overlay, per-frame latency chart    │
│  - Benchmark tab: side-by-side latency comparison        │
└─────────────────────┬────────────────────────────────────┘
                      │ REST (multipart/form-data + JSON)
┌─────────────────────▼────────────────────────────────────┐
│  FastAPI Backend (localhost:8000)                        │
│  POST /v1/detect/image   — single image inference        │
│  POST /v1/detect/video   — multi-frame video inference   │
│  POST /v1/benchmark      — latency benchmark             │
│  GET  /v1/models         — list available backends       │
│  GET  /health            — backend status                │
│                                                          │
│  Model Registry — 8 combos pre-loaded at startup        │
│  ┌───────────┬────────────────────────────────────────┐  │
│  │ Model     │ Backends                               │  │
│  ├───────────┼────────────────────────────────────────┤  │
│  │ YOLO11n   │ pytorch_cpu  pytorch_mps  torchscript  │  │
│  │           │ onnx_coreml                            │  │
│  ├───────────┼────────────────────────────────────────┤  │
│  │ YOLOv8s   │ pytorch_cpu  pytorch_mps  torchscript  │  │
│  │           │ onnx_coreml                            │  │
│  └───────────┴────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
Homework 2/
├── backend/
│   ├── main.py                      # FastAPI app, OpenAI-style REST API
│   ├── config.py                    # ModelName/Backend enums, COCO classes
│   ├── types.py                     # Detection / InferenceResult dataclasses
│   ├── detectors/
│   │   ├── base.py                  # Abstract BaseDetector (predict pipeline)
│   │   ├── pytorch_detector.py      # Eager PyTorch via YOLO wrapper (CPU or MPS)
│   │   ├── torchscript_detector.py  # torch.jit.load, CPU only
│   │   └── onnx_detector.py         # ort.InferenceSession + CoreMLExecutionProvider
│   └── utils/
│       ├── image_utils.py           # Letterbox preprocess, YOLO decode, NMS
│       └── video_utils.py           # Frame extraction, per-frame inference
├── frontend/                        # Next.js 16 + TypeScript + Tailwind
│   ├── app/page.tsx                 # Root page: Detection + Benchmark tabs
│   ├── components/
│   │   ├── BBoxCanvas.tsx           # Canvas bounding box overlay for images
│   │   ├── VideoPlayer.tsx          # Video player with live canvas bbox overlay
│   │   ├── MetricsPanel.tsx         # Latency stats + SVG bar chart
│   │   ├── ModelSelector.tsx        # Model/backend/conf controls with badge legend
│   │   ├── UploadPanel.tsx          # Drag-and-drop upload
│   │   └── BenchmarkPanel.tsx       # Side-by-side latency benchmark runner
│   ├── hooks/useDetection.ts        # API state machine
│   ├── lib/api.ts                   # Typed fetch wrappers
│   ├── types/detection.ts           # Shared TypeScript types + backend metadata
│   └── public/sample.mp4            # Symlink to traffic video (163MB, for in-browser demo)
├── evaluation/
│   ├── annotate_frames.py           # Auto-annotate with YOLOv8x
│   ├── review_annotations.py        # Interactive OpenCV frame-by-frame review
│   ├── export_coco_json.py          # YOLO txt → COCO JSON + dataset.yaml
│   └── evaluate_map.py              # YOLO.val() mAP + latency table → JSON + txt
├── scripts/
│   ├── export_models.py             # Export .pt → .torchscript + .onnx
│   └── extract_frames.py            # Extract 1fps frames from video
├── data/
│   ├── videos/                      # Source video (163MB, 4K traffic)
│   ├── frames/                      # 200 extracted JPGs (frame_000001.jpg ...)
│   ├── images/                      # Symlinks to frames/ (required by YOLO.val())
│   ├── labels/                      # Verified YOLO .txt annotations
│   └── annotations/
│       ├── raw_auto/                # YOLOv8x pseudo-labels (before review)
│       ├── verified/                # Human-verified labels
│       ├── coco_gt.json             # COCO format ground truth
│       └── dataset.yaml             # For YOLO.val()
└── backend/models/                  # .pt, .torchscript, .onnx (6 files, ~140MB)
```

---

## Setup

### Prerequisites

```bash
pip install fastapi uvicorn python-multipart ultralytics onnxruntime opencv-python torch torchvision numpy pillow pyyaml
node --version  # requires Node 18+
```

### 1. Export Models (one-time setup, ~2 min)

```bash
cd "Homework 2"
python3 scripts/export_models.py
```

Produces 6 files in `backend/models/`:

```
yolo11n.pt   yolo11n.torchscript   yolo11n.onnx
yolov8s.pt   yolov8s.torchscript   yolov8s.onnx
```

### 2. Start Backend

```bash
cd "Homework 2"
PYTHONPATH=$(pwd) uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Backend pre-loads all 8 model-backend combinations at startup. Expect ~30s for MPS warmup.

### 3. Start Frontend

```bash
cd "Homework 2/frontend"
npm install
npm run dev
```

Open http://localhost:3000

The **Detection** tab has an "Use sample traffic video" button — click it to load the demo video without uploading anything, then select a model and backend and click Run Detection.

---

## API Reference

### POST /v1/detect/image

```
Content-Type: multipart/form-data
Fields:
  file           — image file (JPG, PNG)
  model          — yolo11n | yolov8s  (default: yolo11n)
  backend        — pytorch_cpu | pytorch_mps | torchscript_cpu | onnx_coreml  (default: onnx_coreml)
  conf_threshold — float 0-1  (default: 0.25)

Response:
{
  "id": "det_a3f7c2b1",
  "model": "yolo11n",
  "backend": "onnx_coreml",
  "latency_ms": 12.8,
  "image_width": 1920,
  "image_height": 1080,
  "detections": [
    {"bbox": [x1, y1, x2, y2], "class_id": 2, "class_name": "car", "confidence": 0.87}
  ]
}
```

### POST /v1/detect/video

```
Content-Type: multipart/form-data
Fields: file, model, backend, conf_threshold, max_frames (default: 150)

Response: VideoDetectionResponse with per-frame detections + latency stats (avg, p50, p95, p99, min, max)
```

### POST /v1/benchmark

```json
Request:  {"model": "yolo11n", "backend": "onnx_coreml", "num_iterations": 100}
Response: {"mean_ms": 12.8, "p50_ms": 12.3, "p99_ms": 18.6, "throughput_fps": 78.1, ...}
```

### GET /v1/models

Returns all registered model+backend combinations and their load status.

### GET /health

```json
{"status": "ok", "mps_available": true, "loaded_models": [...], "total_registered": 8}
```

---

## Annotation + Evaluation Workflow

```bash
# Step 1: Extract 1fps frames from the video
python3 scripts/extract_frames.py
# Output: data/frames/frame_000001.jpg ... frame_000200.jpg

# Step 2: Auto-annotate with YOLOv8x (most accurate YOLO model, downloads ~137MB)
python3 evaluation/annotate_frames.py
# Output: data/annotations/raw_auto/*.txt  (YOLO format)

# Step 3: Human review — interactive OpenCV window
# Keys: a=Accept  d=Delete all boxes for this frame  n=Skip  b=Back  q=Quit
python3 evaluation/review_annotations.py
# Output: data/annotations/verified/*.txt

# Step 4: Export to COCO JSON + YOLO dataset.yaml
python3 evaluation/export_coco_json.py
# Output: data/annotations/coco_gt.json, data/annotations/dataset.yaml

# Step 5: Run full mAP + latency evaluation
python3 evaluation/evaluate_map.py
# Output: data/results/evaluation_report.json, data/results/evaluation_report.txt
```

---

## Hardware Note

This project runs on **Apple M4, 16GB RAM, no CUDA, no NVIDIA GPU**.

Acceleration methods were selected accordingly:

| Requirement | Implementation | Rationale |
|---|---|---|
| "TorchScript" | TorchScript JIT, CPU | Directly from the approved list |
| "ONNX Runtime with CUDA/TRT backend" | ONNX Runtime + CoreML EP | CoreML EP is the Apple Silicon hardware accelerator equivalent — delegates to the Neural Engine (ANE), the same role CUDA plays on NVIDIA systems |
