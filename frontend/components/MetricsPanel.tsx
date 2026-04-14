"use client";

import type { ImageDetectionResponse, VideoDetectionResponse } from "@/types/detection";

interface Props {
  imageResult?: ImageDetectionResponse | null;
  videoResult?: VideoDetectionResponse | null;
}

function StatBadge({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-gray-700 rounded-lg px-3 py-2 flex flex-col items-center">
      <span className="text-xs text-gray-400 uppercase tracking-wide">{label}</span>
      <span className="text-lg font-bold text-white mt-0.5">{value}</span>
    </div>
  );
}

function LatencyBarChart({ latencies }: { latencies: number[] }) {
  if (latencies.length === 0) return null;
  const maxVal = Math.max(...latencies, 1);
  const chartW = 600;
  const chartH = 80;
  const barW = Math.max(2, (chartW / latencies.length) - 1);

  return (
    <div className="mt-4">
      <p className="text-xs text-gray-400 mb-1 uppercase tracking-wide">Per-frame latency (ms)</p>
      <svg
        width="100%"
        viewBox={`0 0 ${chartW} ${chartH}`}
        className="rounded overflow-hidden bg-gray-700"
      >
        {latencies.map((lat, i) => {
          const barH = (lat / maxVal) * chartH;
          const x = i * (chartW / latencies.length);
          const hue = lat > maxVal * 0.8 ? 0 : lat > maxVal * 0.5 ? 40 : 140;
          return (
            <rect
              key={i}
              x={x}
              y={chartH - barH}
              width={barW}
              height={barH}
              fill={`hsl(${hue}, 70%, 50%)`}
              opacity={0.85}
            />
          );
        })}
        {/* Average line */}
        {(() => {
          const avg = latencies.reduce((a, b) => a + b, 0) / latencies.length;
          const y = chartH - (avg / maxVal) * chartH;
          return (
            <line
              x1={0}
              y1={y}
              x2={chartW}
              y2={y}
              stroke="#60a5fa"
              strokeWidth={1}
              strokeDasharray="4 3"
            />
          );
        })()}
      </svg>
      <div className="flex justify-between text-xs text-gray-500 mt-0.5">
        <span>frame 0</span>
        <span className="text-blue-400">— avg</span>
        <span>frame {latencies.length - 1}</span>
      </div>
    </div>
  );
}

export default function MetricsPanel({ imageResult, videoResult }: Props) {
  if (!imageResult && !videoResult) return null;

  if (imageResult) {
    return (
      <div className="bg-gray-800 rounded-xl p-4">
        <h3 className="text-sm font-semibold text-gray-300 mb-3 uppercase tracking-wide">
          Results
        </h3>
        <div className="grid grid-cols-3 gap-2 mb-3">
          <StatBadge label="Latency" value={`${imageResult.latency_ms.toFixed(1)} ms`} />
          <StatBadge label="Detections" value={String(imageResult.detections.length)} />
          <StatBadge label="FPS" value={(1000 / imageResult.latency_ms).toFixed(1)} />
        </div>
        <div className="text-xs text-gray-500 space-y-0.5">
          <div>Model: <span className="text-gray-300">{imageResult.model}</span></div>
          <div>Backend: <span className="text-gray-300">{imageResult.backend}</span></div>
          <div>
            Resolution:{" "}
            <span className="text-gray-300">
              {imageResult.image_width}×{imageResult.image_height}
            </span>
          </div>
          <div>Request ID: <span className="text-gray-500 font-mono">{imageResult.id}</span></div>
        </div>

        {imageResult.detections.length > 0 && (
          <div className="mt-3">
            <p className="text-xs text-gray-400 mb-1 uppercase tracking-wide">Detections</p>
            <div className="max-h-40 overflow-y-auto space-y-1">
              {imageResult.detections.map((d, i) => (
                <div key={i} className="flex justify-between text-xs bg-gray-700 rounded px-2 py-1">
                  <span className="text-white">{d.class_name}</span>
                  <span className="text-gray-400">{(d.confidence * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  }

  // Video result
  const vr = videoResult!;
  const latencies = vr.frames.map((f) => f.latency_ms);

  return (
    <div className="bg-gray-800 rounded-xl p-4">
      <h3 className="text-sm font-semibold text-gray-300 mb-3 uppercase tracking-wide">
        Video Results
      </h3>
      <div className="grid grid-cols-3 gap-2 mb-3">
        <StatBadge label="Avg Latency" value={`${vr.avg_latency_ms.toFixed(1)} ms`} />
        <StatBadge label="Frames" value={String(vr.total_frames_processed)} />
        <StatBadge label="Avg FPS" value={(1000 / vr.avg_latency_ms).toFixed(1)} />
      </div>
      <div className="grid grid-cols-3 gap-2 mb-3">
        <StatBadge label="p50" value={`${vr.p50_latency_ms.toFixed(1)} ms`} />
        <StatBadge label="p95" value={`${vr.p95_latency_ms.toFixed(1)} ms`} />
        <StatBadge label="p99" value={`${vr.p99_latency_ms.toFixed(1)} ms`} />
      </div>

      <div className="text-xs text-gray-500 space-y-0.5 mb-2">
        <div>Model: <span className="text-gray-300">{vr.model}</span></div>
        <div>Backend: <span className="text-gray-300">{vr.backend}</span></div>
      </div>

      <LatencyBarChart latencies={latencies} />
    </div>
  );
}
