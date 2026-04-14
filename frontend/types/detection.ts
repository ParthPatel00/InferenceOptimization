export interface Detection {
  bbox: [number, number, number, number]; // [x1, y1, x2, y2] in pixels
  class_id: number;
  class_name: string;
  confidence: number;
}

export interface FrameResult {
  frame_idx: number;
  latency_ms: number;
  detections: Detection[];
}

export interface ImageDetectionResponse {
  id: string;
  model: string;
  backend: string;
  latency_ms: number;
  image_width: number;
  image_height: number;
  detections: Detection[];
}

export interface VideoDetectionResponse {
  id: string;
  model: string;
  backend: string;
  total_frames_processed: number;
  avg_latency_ms: number;
  p50_latency_ms: number;
  p95_latency_ms: number;
  p99_latency_ms: number;
  min_latency_ms: number;
  max_latency_ms: number;
  frames: FrameResult[];
}

export interface BenchmarkResponse {
  id: string;
  model: string;
  backend: string;
  num_iterations: number;
  mean_ms: number;
  std_ms: number;
  p50_ms: number;
  p95_ms: number;
  p99_ms: number;
  min_ms: number;
  max_ms: number;
  throughput_fps: number;
}

export type ModelName = "yolo11n" | "yolov8s";
export type BackendName =
  | "pytorch_cpu"
  | "pytorch_mps"
  | "torchscript_cpu"
  | "onnx_coreml";

export const MODEL_OPTIONS: { value: ModelName; label: string }[] = [
  { value: "yolo11n", label: "YOLO11n (nano, fast)" },
  { value: "yolov8s", label: "YOLOv8s (small, accurate)" },
];

export const BACKEND_OPTIONS: {
  value: BackendName;
  label: string;
  tag: "baseline" | "accelerated" | "reference";
  desc: string;
}[] = [
  {
    value: "pytorch_cpu",
    label: "PyTorch (CPU)",
    tag: "reference",
    desc: "Unoptimized reference — Python eager mode on CPU",
  },
  {
    value: "pytorch_mps",
    label: "PyTorch (MPS)",
    tag: "baseline",
    desc: "Baseline — Apple Metal GPU via PyTorch eager mode",
  },
  {
    value: "torchscript_cpu",
    label: "TorchScript (CPU)",
    tag: "accelerated",
    desc: "Acceleration #1 — JIT-compiled graph, eliminates Python overhead",
  },
  {
    value: "onnx_coreml",
    label: "ONNX + CoreML (ANE)",
    tag: "accelerated",
    desc: "Acceleration #2 — Apple Neural Engine via ONNX Runtime CoreML EP",
  },
];
