"use client";

import type { ModelName, BackendName } from "@/types/detection";
import { MODEL_OPTIONS, BACKEND_OPTIONS } from "@/types/detection";

interface Props {
  model: ModelName;
  backend: BackendName;
  confThreshold: number;
  onModelChange: (m: ModelName) => void;
  onBackendChange: (b: BackendName) => void;
  onConfChange: (c: number) => void;
  onRun: () => void;
  loading: boolean;
  disabled?: boolean;
}

const TAG_STYLES = {
  reference: "bg-gray-700 text-gray-400",
  baseline: "bg-blue-900/60 text-blue-300",
  accelerated: "bg-green-900/60 text-green-300",
};

const TAG_LABELS = {
  reference: "Reference",
  baseline: "Baseline",
  accelerated: "Accelerated",
};

export default function ModelSelector({
  model,
  backend,
  confThreshold,
  onModelChange,
  onBackendChange,
  onConfChange,
  onRun,
  loading,
  disabled = false,
}: Props) {
  const selectedBackend = BACKEND_OPTIONS.find((o) => o.value === backend);

  return (
    <div className="bg-gray-800 rounded-xl p-4 flex flex-col gap-4">
      {/* Model */}
      <div>
        <label className="block text-xs text-gray-400 mb-1 font-semibold uppercase tracking-wide">
          Model
        </label>
        <select
          value={model}
          onChange={(e) => onModelChange(e.target.value as ModelName)}
          disabled={loading || disabled}
          className="w-full bg-gray-700 text-white text-sm rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
        >
          {MODEL_OPTIONS.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
      </div>

      {/* Backend */}
      <div>
        <label className="block text-xs text-gray-400 mb-1 font-semibold uppercase tracking-wide">
          Backend
        </label>
        <select
          value={backend}
          onChange={(e) => onBackendChange(e.target.value as BackendName)}
          disabled={loading || disabled}
          className="w-full bg-gray-700 text-white text-sm rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
        >
          {BACKEND_OPTIONS.map((o) => (
            <option key={o.value} value={o.value}>
              [{TAG_LABELS[o.tag]}] {o.label}
            </option>
          ))}
        </select>

        {/* Backend description card */}
        {selectedBackend && (
          <div className="mt-2 rounded-lg bg-gray-700/50 border border-gray-600 px-3 py-2 flex items-start gap-2">
            <span
              className={`mt-0.5 shrink-0 text-[10px] font-bold uppercase px-1.5 py-0.5 rounded ${TAG_STYLES[selectedBackend.tag]}`}
            >
              {TAG_LABELS[selectedBackend.tag]}
            </span>
            <p className="text-xs text-gray-300 leading-relaxed">{selectedBackend.desc}</p>
          </div>
        )}
      </div>

      {/* Confidence threshold */}
      <div>
        <label className="block text-xs text-gray-400 mb-1 font-semibold uppercase tracking-wide">
          Confidence Threshold: {(confThreshold * 100).toFixed(0)}%
        </label>
        <input
          type="range"
          min={0.1}
          max={0.9}
          step={0.05}
          value={confThreshold}
          onChange={(e) => onConfChange(Number(e.target.value))}
          disabled={loading || disabled}
          className="w-full accent-blue-500 disabled:opacity-50"
        />
        <div className="flex justify-between text-xs text-gray-500 mt-0.5">
          <span>10%</span>
          <span>90%</span>
        </div>
      </div>

      {/* Run button */}
      <button
        onClick={onRun}
        disabled={loading || disabled}
        className="w-full py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-semibold rounded-lg transition-colors"
      >
        {loading ? (
          <span className="flex items-center justify-center gap-2">
            <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"/>
            </svg>
            Running...
          </span>
        ) : (
          "Run Detection"
        )}
      </button>

      {/* Legend */}
      <div className="border-t border-gray-700 pt-3 flex flex-wrap gap-2">
        {(["reference", "baseline", "accelerated"] as const).map((tag) => (
          <span
            key={tag}
            className={`text-[10px] font-bold uppercase px-1.5 py-0.5 rounded ${TAG_STYLES[tag]}`}
          >
            {TAG_LABELS[tag]}
          </span>
        ))}
        <span className="text-[10px] text-gray-500 self-center">
          — select a backend to see details
        </span>
      </div>
    </div>
  );
}
