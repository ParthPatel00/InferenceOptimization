import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Object Detection — CMPE258",
  description: "YOLO inference with TorchScript and ONNX/CoreML acceleration",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full">
      <body className="min-h-full bg-gray-900 text-gray-100">{children}</body>
    </html>
  );
}
