"use client";

import { useEffect, useRef } from "react";
import type { Detection } from "@/types/detection";

interface Props {
  imageSrc: string;
  detections: Detection[];
  imageWidth: number;
  imageHeight: number;
  className?: string;
}

function classColor(classId: number): string {
  const hue = (classId * 47) % 360;
  return `hsl(${hue}, 70%, 50%)`;
}

function classColorRGBA(classId: number, alpha = 1): string {
  const hue = (classId * 47) % 360;
  return `hsla(${hue}, 70%, 50%, ${alpha})`;
}

export default function BBoxCanvas({
  imageSrc,
  detections,
  imageWidth,
  imageHeight,
  className = "",
}: Props) {
  const imgRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img) return;

    const draw = () => {
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      // Match canvas size to displayed image size
      const rect = img.getBoundingClientRect();
      canvas.width = rect.width;
      canvas.height = rect.height;

      const scaleX = rect.width / imageWidth;
      const scaleY = rect.height / imageHeight;

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      for (const det of detections) {
        const [x1, y1, x2, y2] = det.bbox;
        const sx1 = x1 * scaleX;
        const sy1 = y1 * scaleY;
        const sw = (x2 - x1) * scaleX;
        const sh = (y2 - y1) * scaleY;

        // Box
        ctx.strokeStyle = classColor(det.class_id);
        ctx.lineWidth = 2;
        ctx.strokeRect(sx1, sy1, sw, sh);

        // Label background
        const label = `${det.class_name} ${(det.confidence * 100).toFixed(0)}%`;
        ctx.font = "bold 12px sans-serif";
        const textW = ctx.measureText(label).width;
        const textH = 14;
        ctx.fillStyle = classColorRGBA(det.class_id, 0.85);
        ctx.fillRect(sx1, sy1 - textH - 2, textW + 6, textH + 2);

        // Label text
        ctx.fillStyle = "#fff";
        ctx.fillText(label, sx1 + 3, sy1 - 3);
      }
    };

    if (img.complete) {
      draw();
    } else {
      img.addEventListener("load", draw);
      return () => img.removeEventListener("load", draw);
    }
  }, [detections, imageSrc, imageWidth, imageHeight]);

  return (
    <div className={`relative inline-block ${className}`}>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        ref={imgRef}
        src={imageSrc}
        alt="Detection result"
        className="block max-w-full max-h-[70vh] object-contain"
      />
      <canvas
        ref={canvasRef}
        className="absolute inset-0 pointer-events-none"
        style={{ width: "100%", height: "100%" }}
      />
    </div>
  );
}
