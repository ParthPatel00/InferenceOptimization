"use client";

import { useState } from "react";
import type { ModelName, BackendName, BenchmarkResponse } from "@/types/detection";
import { MODEL_OPTIONS, BACKEND_OPTIONS } from "@/types/detection";
import { runBenchmark } from "@/lib/api";

const TAG_STYLES = {
  reference: "bg-gray-700 text-gray-400",
  baseline: "bg-blue-900/60 text-blue-300",
  accelerated: "bg-green-900/60 text-green-300",
};

const TAG_LABELS = {
  reference: "Ref",
  baseline: "Baseline",
  accelerated: "Accel",
};

function backendTag(backendValue: string) {
  const opt = BACKEND_OPTIONS.find((o) => o.value === backendValue);
  if (!opt) return null;
  return (
    <span className={`text-[10px] font-bold uppercase px-1 py-0.5 rounded ${TAG_STYLES[opt.tag]}`}>
      {TAG_LABELS[opt.tag]}
    </span>
  );
}

export default function BenchmarkPanel() {
  const [model, setModel] = useState<ModelName>("yolo11n");
  const [backend, setBackend] = useState<BackendName>("pytorch_cpu");
  const [iterations, setIterations] = useState(100);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<BenchmarkResponse[]>([]);
  const [error, setError] = useState<string | null>(null);

  const handleBenchmark = async (singleModel?: ModelName, singleBackend?: BackendName) => {
    setLoading(true);
    setError(null);

    const configs: [ModelName, BackendName][] = singleModel && singleBackend
      ? [[singleModel, singleBackend]]
      : MODEL_OPTIONS.flatMap((m) =>
          BACKEND_OPTIONS.map((b) => [m.value as ModelName, b.value as BackendName] as [ModelName, BackendName])
        );

    const newResults: BenchmarkResponse[] = [];
    for (const [m, b] of configs) {
      try {
        const r = await runBenchmark(m, b, iterations);
        newResults.push(r);
      } catch (e) {
        console.error(`Benchmark failed for ${m}/${b}:`, e);
      }
    }

    setResults((prev) => {
      const updated = [...prev];
      for (const r of newResults) {
        const idx = updated.findIndex((x) => x.model === r.model && x.backend === r.backend);
        if (idx >= 0) updated[idx] = r;
        else updated.push(r);
      }
      return updated;
    });

    setLoading(false);
  };

  return (
    <div className="space-y-5">
      {/* Explanation card */}
      <div className="bg-gray-800 rounded-xl p-4 space-y-3 text-sm text-gray-300">
        <h3 className="font-semibold text-white">Inference Acceleration Comparison</h3>
        <p className="text-xs leading-relaxed text-gray-400">
          This benchmark compares four execution backends for the same YOLO model on Apple M4.
          The goal is to measure the speedup of two hardware-accelerated backends relative to the
          unoptimized CPU baseline.
        </p>
        <div className="grid grid-cols-1 gap-2 text-xs">
          {BACKEND_OPTIONS.map((o) => (
            <div key={o.value} className="flex items-start gap-2 bg-gray-700/40 rounded-lg px-3 py-2">
              <span className={`mt-0.5 shrink-0 text-[10px] font-bold uppercase px-1.5 py-0.5 rounded ${TAG_STYLES[o.tag]}`}>
                {TAG_LABELS[o.tag]}
              </span>
              <div>
                <span className="text-white font-medium">{o.label}</span>
                <span className="text-gray-400 ml-2">{o.desc}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Controls */}
      <div className="bg-gray-800 rounded-xl p-4 space-y-3">
        <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wide">
          Run Benchmark
        </h3>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="block text-xs text-gray-400 mb-1">Model</label>
            <select
              value={model}
              onChange={(e) => setModel(e.target.value as ModelName)}
              disabled={loading}
              className="w-full bg-gray-700 text-white text-sm rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {MODEL_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">Backend</label>
            <select
              value={backend}
              onChange={(e) => setBackend(e.target.value as BackendName)}
              disabled={loading}
              className="w-full bg-gray-700 text-white text-sm rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {BACKEND_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>
                  [{TAG_LABELS[o.tag]}] {o.label}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div>
          <label className="block text-xs text-gray-400 mb-1">
            Iterations: {iterations}
          </label>
          <input
            type="range"
            min={10}
            max={300}
            step={10}
            value={iterations}
            onChange={(e) => setIterations(Number(e.target.value))}
            disabled={loading}
            className="w-full accent-blue-500"
          />
        </div>

        <div className="flex gap-2">
          <button
            onClick={() => handleBenchmark(model, backend)}
            disabled={loading}
            className="flex-1 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-600 text-white font-semibold rounded-lg transition-colors text-sm"
          >
            {loading ? "Running..." : "Run Single"}
          </button>
          <button
            onClick={() => handleBenchmark()}
            disabled={loading}
            className="flex-1 py-2 bg-purple-700 hover:bg-purple-600 disabled:bg-gray-600 text-white font-semibold rounded-lg transition-colors text-sm"
          >
            {loading ? "Running..." : "Run All Combos"}
          </button>
        </div>

        {error && (
          <p className="text-red-400 text-sm bg-red-900/20 rounded px-3 py-2">{error}</p>
        )}
      </div>

      {/* Results table */}
      {results.length > 0 && (
        <div className="bg-gray-800 rounded-xl p-4 overflow-x-auto">
          <h3 className="text-sm font-semibold text-gray-300 mb-1 uppercase tracking-wide">
            Results — sorted fastest to slowest
          </h3>
          <p className="text-xs text-gray-500 mb-3">
            Lower latency = faster. FPS = 1000 / mean_ms. Accelerated backends should be faster than the CPU baseline.
          </p>
          <table className="w-full text-xs text-left">
            <thead>
              <tr className="text-gray-400 border-b border-gray-600">
                <th className="pb-2 pr-2">Type</th>
                <th className="pb-2 pr-3">Model</th>
                <th className="pb-2 pr-3">Backend</th>
                <th className="pb-2 pr-3 text-right">Mean</th>
                <th className="pb-2 pr-3 text-right">p50</th>
                <th className="pb-2 pr-3 text-right">p99</th>
                <th className="pb-2 text-right">FPS</th>
              </tr>
            </thead>
            <tbody>
              {results
                .slice()
                .sort((a, b) => a.mean_ms - b.mean_ms)
                .map((r) => (
                  <tr
                    key={`${r.model}-${r.backend}`}
                    className="border-b border-gray-700 hover:bg-gray-700/40"
                  >
                    <td className="py-1.5 pr-2">{backendTag(r.backend)}</td>
                    <td className="py-1.5 pr-3 text-white">{r.model}</td>
                    <td className="py-1.5 pr-3 text-gray-300">{r.backend}</td>
                    <td className="py-1.5 pr-3 text-right font-mono">{r.mean_ms.toFixed(1)} ms</td>
                    <td className="py-1.5 pr-3 text-right font-mono text-gray-400">{r.p50_ms.toFixed(1)}</td>
                    <td className="py-1.5 pr-3 text-right font-mono text-gray-400">{r.p99_ms.toFixed(1)}</td>
                    <td className="py-1.5 text-right font-mono text-green-400">
                      {r.throughput_fps.toFixed(1)}
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
