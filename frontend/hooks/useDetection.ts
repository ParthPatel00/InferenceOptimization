"use client";

import { useState } from "react";
import type {
  ImageDetectionResponse,
  VideoDetectionResponse,
  ModelName,
  BackendName,
} from "@/types/detection";
import { detectImage, detectVideo } from "@/lib/api";

type Status = "idle" | "loading" | "success" | "error";

export function useDetection() {
  const [status, setStatus] = useState<Status>("idle");
  const [imageResult, setImageResult] = useState<ImageDetectionResponse | null>(null);
  const [videoResult, setVideoResult] = useState<VideoDetectionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const reset = () => {
    setStatus("idle");
    setImageResult(null);
    setVideoResult(null);
    setError(null);
  };

  const runImageDetection = async (
    file: File,
    model: ModelName,
    backend: BackendName,
    conf: number
  ) => {
    setStatus("loading");
    setError(null);
    setImageResult(null);
    setVideoResult(null);
    try {
      const result = await detectImage(file, model, backend, conf);
      setImageResult(result);
      setStatus("success");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus("error");
    }
  };

  const runVideoDetection = async (
    file: File,
    model: ModelName,
    backend: BackendName,
    conf: number,
    maxFrames: number = 150
  ) => {
    setStatus("loading");
    setError(null);
    setImageResult(null);
    setVideoResult(null);
    try {
      const result = await detectVideo(file, model, backend, conf, maxFrames);
      setVideoResult(result);
      setStatus("success");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus("error");
    }
  };

  return {
    status,
    imageResult,
    videoResult,
    error,
    reset,
    runImageDetection,
    runVideoDetection,
  };
}
