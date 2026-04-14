"""
base.py — Abstract base class for all detectors plus shared dataclasses.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod

import numpy as np

from backend.types import Detection, InferenceResult
from backend.utils.image_utils import preprocess_image, postprocess_detections
from backend.config import DEFAULT_CONF_THRESHOLD, DEFAULT_IOU_THRESHOLD


class BaseDetector(ABC):
    """
    Abstract detector. Subclasses implement `load()` and `_infer_raw()`.
    The `predict()` method handles the full preprocess → infer → postprocess pipeline
    and measures wall-clock latency.
    """

    def __init__(
        self,
        model_name: str,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    ) -> None:
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load model weights and perform warmup. Must set self._loaded = True."""
        ...

    @abstractmethod
    def _infer_raw(self, img_nchw: np.ndarray) -> np.ndarray:
        """
        Run forward pass on a preprocessed float32 NCHW tensor [1, 3, 640, 640].
        Return raw YOLO output of shape [1, 84, 8400].
        """
        ...

    def predict(
        self,
        image_bgr: np.ndarray,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
    ) -> InferenceResult:
        """Full pipeline: preprocess → infer → postprocess. Returns latency in ms."""
        if not self._loaded:
            self.load()

        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        iou = iou_threshold if iou_threshold is not None else self.iou_threshold

        orig_h, orig_w = image_bgr.shape[:2]

        # Preprocess
        img_nchw, scale, pad = preprocess_image(image_bgr)

        # Timed inference
        t0 = time.perf_counter()
        raw_output = self._infer_raw(img_nchw)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Postprocess
        detections = postprocess_detections(
            raw_output=raw_output,
            scale=scale,
            pad=pad,
            orig_shape=(orig_h, orig_w),
            conf_threshold=conf,
            iou_threshold=iou,
        )

        return InferenceResult(
            detections=detections,
            latency_ms=latency_ms,
            image_width=orig_w,
            image_height=orig_h,
        )
