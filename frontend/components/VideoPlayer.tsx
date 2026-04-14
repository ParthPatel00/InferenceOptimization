"use client";

import { useEffect, useRef, useState } from "react";
import type { VideoDetectionResponse, FrameResult } from "@/types/detection";
import BBoxCanvas from "./BBoxCanvas";

interface Props {
  videoFile: File;
  result: VideoDetectionResponse;
}

export default function VideoPlayer({ videoFile, result }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const snapshotCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const [videoSrc, setVideoSrc] = useState<string>("");
  const [currentFrame, setCurrentFrame] = useState<FrameResult | null>(
    result.frames[0] ?? null
  );
  const [frameSrc, setFrameSrc] = useState<string>("");
  const [videoWidth, setVideoWidth] = useState(1920);
  const [videoHeight, setVideoHeight] = useState(1080);
  const [hasInteracted, setHasInteracted] = useState(false);

  // Create object URL for the video file
  useEffect(() => {
    const url = URL.createObjectURL(videoFile);
    setVideoSrc(url);
    return () => URL.revokeObjectURL(url);
  }, [videoFile]);

  // Capture the current video frame as a JPEG data URL
  const captureFrame = () => {
    const video = videoRef.current;
    if (!video || video.videoWidth === 0) return;

    if (!snapshotCanvasRef.current) {
      snapshotCanvasRef.current = document.createElement("canvas");
    }
    const canvas = snapshotCanvasRef.current;
    if (canvas.width !== video.videoWidth) canvas.width = video.videoWidth;
    if (canvas.height !== video.videoHeight) canvas.height = video.videoHeight;
    canvas.getContext("2d")?.drawImage(video, 0, 0);
    setFrameSrc(canvas.toDataURL("image/jpeg", 0.85));
  };

  // On video metadata load, record dimensions and capture first frame
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const onLoadedMetadata = () => {
      setVideoWidth(video.videoWidth);
      setVideoHeight(video.videoHeight);
    };

    // Seek slightly into the video so the first frame is decodable
    const onCanPlay = () => {
      if (!hasInteracted) {
        video.currentTime = 0.01;
      }
    };

    const onSeeked = () => {
      captureFrame();
    };

    video.addEventListener("loadedmetadata", onLoadedMetadata);
    video.addEventListener("canplay", onCanPlay);
    video.addEventListener("seeked", onSeeked);
    return () => {
      video.removeEventListener("loadedmetadata", onLoadedMetadata);
      video.removeEventListener("canplay", onCanPlay);
      video.removeEventListener("seeked", onSeeked);
    };
  }, [hasInteracted]);

  // Map playback time to frame index and capture snapshot
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const onTimeUpdate = () => {
      if (result.frames.length === 0) return;
      const progress = video.currentTime / (video.duration || 1);
      const idx = Math.min(
        Math.floor(progress * result.frames.length),
        result.frames.length - 1
      );
      setCurrentFrame(result.frames[idx] ?? null);
      captureFrame();
      setHasInteracted(true);
    };

    const onPlay = () => setHasInteracted(true);

    video.addEventListener("timeupdate", onTimeUpdate);
    video.addEventListener("play", onPlay);
    return () => {
      video.removeEventListener("timeupdate", onTimeUpdate);
      video.removeEventListener("play", onPlay);
    };
  }, [result.frames]);

  const jumpToFrame = (frame: FrameResult) => {
    setCurrentFrame(frame);
    setHasInteracted(true);
    const video = videoRef.current;
    if (video) {
      video.currentTime =
        (frame.frame_idx / result.frames.length) * (video.duration || 0);
    }
  };

  return (
    <div className="space-y-3">
      {/* Plain video player — no overlay */}
      <div className="relative bg-black rounded-lg overflow-hidden">
        <video
          ref={videoRef}
          src={videoSrc || undefined}
          controls
          className="w-full max-h-[40vh] block"
          playsInline
        />
        {!hasInteracted && (
          <div className="absolute top-2 inset-x-0 flex justify-center pointer-events-none">
            <span className="bg-black/70 text-white text-xs px-3 py-1.5 rounded-full">
              Play or scrub — the frame below will update with detections
            </span>
          </div>
        )}
      </div>

      {/* Current frame with bounding boxes */}
      {currentFrame && frameSrc && (
        <div>
          <p className="text-xs text-gray-400 mb-1.5 flex gap-4">
            <span className="text-white font-medium">Frame {currentFrame.frame_idx}</span>
            <span>{currentFrame.latency_ms.toFixed(1)} ms inference</span>
            <span>{currentFrame.detections.length} detection(s)</span>
          </p>
          <BBoxCanvas
            imageSrc={frameSrc}
            detections={currentFrame.detections}
            imageWidth={videoWidth}
            imageHeight={videoHeight}
            className="w-full rounded-lg overflow-hidden"
          />
        </div>
      )}

      {/* Frame list */}
      <div className="bg-gray-700 rounded-lg p-3">
        <p className="text-xs text-gray-400 mb-2 uppercase tracking-wide">
          {result.frames.length} frames processed — click any row to jump
        </p>
        <div className="max-h-36 overflow-y-auto space-y-1">
          {result.frames.map((frame) => (
            <button
              key={frame.frame_idx}
              onClick={() => jumpToFrame(frame)}
              className={`w-full flex justify-between text-xs rounded px-2 py-1 transition-colors ${
                currentFrame?.frame_idx === frame.frame_idx
                  ? "bg-blue-600 text-white"
                  : "bg-gray-800 hover:bg-gray-600 text-gray-300"
              }`}
            >
              <span>Frame {frame.frame_idx}</span>
              <span>{frame.detections.length} obj</span>
              <span>{frame.latency_ms.toFixed(1)} ms</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
