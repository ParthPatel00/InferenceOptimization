"use client";

import { useCallback, useRef, useState } from "react";

interface Props {
  onFileSelected: (file: File, type: "image" | "video") => void;
  disabled?: boolean;
}

export default function UploadPanel({ onFileSelected, disabled = false }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const handleFile = (file: File) => {
    const type = file.type.startsWith("video/") ? "video" : "image";
    onFileSelected(file, type);
  };

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      if (disabled) return;
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [disabled]
  );

  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    if (!disabled) setDragging(true);
  };

  const onDragLeave = () => setDragging(false);

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  return (
    <div
      onClick={() => !disabled && inputRef.current?.click()}
      onDrop={onDrop}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      className={`
        border-2 border-dashed rounded-xl p-8 flex flex-col items-center justify-center
        cursor-pointer transition-all min-h-[140px] select-none
        ${dragging ? "border-blue-400 bg-blue-900/20" : "border-gray-600 hover:border-gray-400"}
        ${disabled ? "opacity-50 cursor-not-allowed" : ""}
      `}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*,video/*"
        className="hidden"
        onChange={onInputChange}
        disabled={disabled}
      />
      <svg
        className="w-10 h-10 text-gray-500 mb-3"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={1.5}
          d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
        />
      </svg>
      <p className="text-gray-400 text-sm text-center">
        <span className="font-semibold text-white">Click to upload</span> or drag and drop
      </p>
      <p className="text-gray-500 text-xs mt-1">Images (JPG, PNG) or Videos (MP4, MOV)</p>
    </div>
  );
}
