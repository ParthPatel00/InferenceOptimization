"""
pytorch_detector.py — PyTorch eager inference detector (CPU or MPS).

Uses the Ultralytics YOLO wrapper's predict pipeline, which correctly handles
preprocessing (letterbox, normalize) and postprocessing (decode, NMS) on both
CPU and MPS (Apple Metal GPU).

Why not use yolo.model directly?
  Calling `yolo.model.eval().to('mps')` and running a forward pass bypasses
  the wrapper's internal preprocessing — the raw DetectionModel output on MPS
  produces near-zero confidence values (~0.008 max vs ~0.83 on CPU). Using the
  full wrapper's predict() path avoids this and works correctly on all devices.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from backend.detectors.base import BaseDetector
from backend.types import Detection, InferenceResult
from backend.config import MODELS_DIR, COCO_CLASSES, DEFAULT_CONF_THRESHOLD, DEFAULT_IOU_THRESHOLD


class PyTorchDetector(BaseDetector):
    """
    PyTorch eager-mode detector.
    device: "mps" for Apple Metal GPU, "cpu" for CPU.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "mps",
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    ) -> None:
        super().__init__(model_name, conf_threshold, iou_threshold)
        # Resolve actual device (fall back to cpu if MPS not available)
        if device == "mps" and not torch.backends.mps.is_available():
            device = "cpu"
        self._device_str = device
        self.yolo = None  # Ultralytics YOLO wrapper

    def load(self) -> None:
        from ultralytics import YOLO

        pt_path = MODELS_DIR / f"{self.model_name}.pt"
        if not pt_path.exists():
            # Auto-download via ultralytics
            self.yolo = YOLO(f"{self.model_name}.pt")
        else:
            self.yolo = YOLO(str(pt_path))

        # Warmup — suppress first-run compilation overhead
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(3):
            self.yolo.predict(
                dummy,
                device=self._device_str,
                verbose=False,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
            )

        self._loaded = True

    def _infer_raw(self, img_nchw: np.ndarray) -> np.ndarray:
        # Not used — predict() is overridden directly below
        raise NotImplementedError("PyTorchDetector overrides predict() directly")

    def predict(
        self,
        image_bgr: np.ndarray,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
    ) -> InferenceResult:
        """
        Full inference pipeline using the YOLO wrapper.
        Times only the predict call (includes preprocessing + inference + NMS).
        """
        if not self._loaded:
            self.load()

        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        iou = iou_threshold if iou_threshold is not None else self.iou_threshold

        orig_h, orig_w = image_bgr.shape[:2]

        t0 = time.perf_counter()
        results = self.yolo.predict(
            image_bgr,
            device=self._device_str,
            verbose=False,
            conf=conf,
            iou=iou,
        )
        # MPS ops are submitted to a command queue asynchronously; synchronize
        # before reading results back to CPU.
        if self._device_str == "mps":
            torch.mps.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        detections: list[Detection] = []
        if results and results[0].boxes is not None and len(results[0].boxes):
            boxes = results[0].boxes
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            for i in range(len(xyxy)):
                cid = int(cls_ids[i])
                detections.append(
                    Detection(
                        bbox=[
                            float(xyxy[i, 0]),
                            float(xyxy[i, 1]),
                            float(xyxy[i, 2]),
                            float(xyxy[i, 3]),
                        ],
                        class_id=cid,
                        class_name=COCO_CLASSES.get(cid, f"class_{cid}"),
                        confidence=float(confs[i]),
                    )
                )

        return InferenceResult(
            detections=detections,
            latency_ms=latency_ms,
            image_width=orig_w,
            image_height=orig_h,
        )
