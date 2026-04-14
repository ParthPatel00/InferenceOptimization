"use client";

import { useState, useEffect } from "react";
import type { ModelName, BackendName } from "@/types/detection";
import { useDetection } from "@/hooks/useDetection";
import UploadPanel from "@/components/UploadPanel";
import ModelSelector from "@/components/ModelSelector";
import BBoxCanvas from "@/components/BBoxCanvas";
import VideoPlayer from "@/components/VideoPlayer";
import MetricsPanel from "@/components/MetricsPanel";
import BenchmarkPanel from "@/components/BenchmarkPanel";
import { getHealth } from "@/lib/api";

type Tab = "detection" | "benchmark";

export default function Home() {
  const [activeTab, setActiveTab] = useState<Tab>("detection");
  const [file, setFile] = useState<File | null>(null);
  const [fileType, setFileType] = useState<"image" | "video">("image");
  const [previewSrc, setPreviewSrc] = useState<string>("");
  const [model, setModel] = useState<ModelName>("yolo11n");
  const [backend, setBackend] = useState<BackendName>("onnx_coreml");
  const [conf, setConf] = useState(0.25);
  const [backendStatus, setBackendStatus] = useState<"unknown" | "ok" | "error">("unknown");
  const [loadingSample, setLoadingSample] = useState(false);

  const { status, imageResult, videoResult, error, runImageDetection, runVideoDetection, reset } =
    useDetection();

  // Check backend health on mount
  useEffect(() => {
    getHealth()
      .then(() => setBackendStatus("ok"))
      .catch(() => setBackendStatus("error"));
  }, []);

  const handleFileSelected = (selectedFile: File, type: "image" | "video") => {
    setFile(selectedFile);
    setFileType(type);
    reset();
    if (type === "image") {
      const url = URL.createObjectURL(selectedFile);
      setPreviewSrc(url);
    } else {
      setPreviewSrc("");
    }
  };

  const handleUseSampleVideo = async () => {
    setLoadingSample(true);
    try {
      const res = await fetch("/sample.mp4");
      const blob = await res.blob();
      const sampleFile = new File([blob], "sample.mp4", { type: "video/mp4" });
      handleFileSelected(sampleFile, "video");
    } catch {
      console.error("Failed to load sample video");
    } finally {
      setLoadingSample(false);
    }
  };

  const handleRun = () => {
    if (!file) return;
    if (fileType === "image") {
      runImageDetection(file, model, backend, conf);
    } else {
      runVideoDetection(file, model, backend, conf, 150);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100">
      {/* Header */}
      <header className="border-b border-gray-800 px-6 py-4 flex items-center justify-between">
        <div>
          <h1 className="text-lg font-bold tracking-tight">Object Detection</h1>
          <p className="text-xs text-gray-500">CMPE258 HW2 — YOLO11n & YOLOv8s on Apple M4</p>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <span
            className={`w-2 h-2 rounded-full ${
              backendStatus === "ok"
                ? "bg-green-500"
                : backendStatus === "error"
                ? "bg-red-500"
                : "bg-yellow-500"
            }`}
          />
          <span className="text-gray-400">
            {backendStatus === "ok"
              ? "Backend connected"
              : backendStatus === "error"
              ? "Backend offline"
              : "Connecting..."}
          </span>
        </div>
      </header>

      {/* Tabs */}
      <div className="border-b border-gray-800 px-6">
        <div className="flex gap-1">
          {(["detection", "benchmark"] as Tab[]).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-2.5 text-sm font-medium capitalize transition-colors border-b-2 ${
                activeTab === tab
                  ? "border-blue-500 text-white"
                  : "border-transparent text-gray-500 hover:text-gray-300"
              }`}
            >
              {tab}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <main className="max-w-7xl mx-auto px-6 py-6">
        {activeTab === "detection" ? (
          <div className="grid grid-cols-1 lg:grid-cols-[360px_1fr] gap-6">
            {/* Left panel */}
            <div className="space-y-4">
              <UploadPanel
                onFileSelected={handleFileSelected}
                disabled={status === "loading"}
              />

              {/* Sample video shortcut */}
              <div className="flex items-center gap-2">
                <div className="flex-1 border-t border-gray-700" />
                <span className="text-xs text-gray-500 shrink-0">or use built-in demo</span>
                <div className="flex-1 border-t border-gray-700" />
              </div>
              <button
                onClick={handleUseSampleVideo}
                disabled={loadingSample || status === "loading"}
                className="w-full flex items-center justify-center gap-2 py-2 px-3 bg-gray-700 hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed text-sm text-gray-200 rounded-lg transition-colors border border-gray-600"
              >
                {loadingSample ? (
                  <>
                    <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"/>
                    </svg>
                    Loading video...
                  </>
                ) : (
                  <>
                    <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 10l4.553-2.276A1 1 0 0121 8.68v6.64a1 1 0 01-1.447.894L15 14M3 8a2 2 0 012-2h8a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V8z" />
                    </svg>
                    Use sample traffic video
                  </>
                )}
              </button>

              {file && (
                <div className="text-xs text-gray-500 bg-gray-800 rounded-lg px-3 py-2 flex justify-between">
                  <span className="truncate">{file.name}</span>
                  <span className="ml-2 text-gray-600 shrink-0">
                    {(file.size / 1e6).toFixed(1)} MB
                  </span>
                </div>
              )}

              <ModelSelector
                model={model}
                backend={backend}
                confThreshold={conf}
                onModelChange={setModel}
                onBackendChange={setBackend}
                onConfChange={setConf}
                onRun={handleRun}
                loading={status === "loading"}
                disabled={!file}
              />

              {error && (
                <div className="bg-red-900/30 border border-red-700 rounded-lg p-3 text-sm text-red-300">
                  <p className="font-semibold mb-1">Error</p>
                  <p className="text-xs">{error}</p>
                </div>
              )}

              <MetricsPanel imageResult={imageResult} videoResult={videoResult} />
            </div>

            {/* Right panel — results */}
            <div className="min-h-[400px] bg-gray-800 rounded-xl flex items-center justify-center overflow-hidden">
              {status === "loading" && (
                <div className="flex flex-col items-center gap-3 text-gray-400">
                  <svg className="animate-spin h-10 w-10 text-blue-500" viewBox="0 0 24 24" fill="none">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"/>
                  </svg>
                  <p className="text-sm">Running inference...</p>
                </div>
              )}

              {status === "idle" && !file && (
                <p className="text-gray-600 text-sm">Upload a file or use the sample video to get started</p>
              )}

              {status === "idle" && file && fileType === "image" && previewSrc && (
                /* eslint-disable-next-line @next/next/no-img-element */
                <img src={previewSrc} alt="Preview" className="max-w-full max-h-[70vh] object-contain" />
              )}

              {status === "success" && imageResult && previewSrc && (
                <BBoxCanvas
                  imageSrc={previewSrc}
                  detections={imageResult.detections}
                  imageWidth={imageResult.image_width}
                  imageHeight={imageResult.image_height}
                />
              )}

              {status === "success" && videoResult && file && (
                <div className="w-full p-4">
                  <VideoPlayer videoFile={file} result={videoResult} />
                </div>
              )}

              {status === "idle" && file && fileType === "video" && (
                <div className="text-center text-gray-500 text-sm p-8">
                  <p className="text-2xl mb-2">🎬</p>
                  <p>{file.name}</p>
                  <p className="text-xs mt-1">Click Run Detection to process</p>
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className="max-w-3xl mx-auto">
            <BenchmarkPanel />
          </div>
        )}
      </main>
    </div>
  );
}
