import type {
  ImageDetectionResponse,
  VideoDetectionResponse,
  BenchmarkResponse,
  ModelName,
  BackendName,
} from "@/types/detection";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function throwOnError(res: Response): Promise<Response> {
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API error ${res.status}: ${text}`);
  }
  return res;
}

export async function detectImage(
  file: File,
  model: ModelName,
  backend: BackendName,
  confThreshold: number = 0.25
): Promise<ImageDetectionResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("model", model);
  form.append("backend", backend);
  form.append("conf_threshold", String(confThreshold));
  const res = await fetch(`${API_BASE}/v1/detect/image`, {
    method: "POST",
    body: form,
  });
  await throwOnError(res);
  return res.json();
}

export async function detectVideo(
  file: File,
  model: ModelName,
  backend: BackendName,
  confThreshold: number = 0.25,
  maxFrames: number = 150
): Promise<VideoDetectionResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("model", model);
  form.append("backend", backend);
  form.append("conf_threshold", String(confThreshold));
  form.append("max_frames", String(maxFrames));
  const res = await fetch(`${API_BASE}/v1/detect/video`, {
    method: "POST",
    body: form,
  });
  await throwOnError(res);
  return res.json();
}

export async function runBenchmark(
  model: ModelName,
  backend: BackendName,
  numIterations: number = 100
): Promise<BenchmarkResponse> {
  const res = await fetch(`${API_BASE}/v1/benchmark`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model, backend, num_iterations: numIterations }),
  });
  await throwOnError(res);
  return res.json();
}

export async function getHealth(): Promise<{
  status: string;
  mps_available: boolean;
  loaded_models: string[];
}> {
  const res = await fetch(`${API_BASE}/health`);
  await throwOnError(res);
  return res.json();
}
