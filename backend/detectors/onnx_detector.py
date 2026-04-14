"""
onnx_detector.py — ONNX Runtime detector with CoreML execution provider.
On Apple Silicon, CoreMLExecutionProvider delegates computation to the
Neural Engine (ANE) or Metal GPU — the Apple equivalent of ONNX+CUDA.

Satisfies the 'ONNX Runtime with [hardware] backend' acceleration requirement.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort

from backend.detectors.base import BaseDetector
from backend.config import MODELS_DIR, DEFAULT_CONF_THRESHOLD, DEFAULT_IOU_THRESHOLD


class ONNXCoreMLDetector(BaseDetector):
    """
    ONNX Runtime + CoreML Execution Provider detector.
    Falls back to CPU if CoreML is unavailable.
    """

    def __init__(
        self,
        model_name: str,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    ) -> None:
        super().__init__(model_name, conf_threshold, iou_threshold)
        self.session: ort.InferenceSession | None = None
        self.input_name: str = ""
        self.active_providers: list[str] = []

    def load(self) -> None:
        onnx_path = MODELS_DIR / f"{self.model_name}.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {onnx_path}\n"
                "Run: python3 scripts/export_models.py"
            )

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4

        # Request CoreML first; ORT will fall back to CPU if unsupported
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]

        self.session = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_options,
            providers=providers,
        )
        self.active_providers = self.session.get_providers()
        self.input_name = self.session.get_inputs()[0].name

        # Warmup
        dummy = np.zeros((1, 3, 640, 640), dtype=np.float32)
        for _ in range(5):
            self.session.run(None, {self.input_name: dummy})

        using_coreml = "CoreMLExecutionProvider" in self.active_providers
        print(
            f"[ONNXCoreMLDetector:{self.model_name}] providers={self.active_providers} "
            f"(CoreML active: {using_coreml})"
        )

        self._loaded = True

    def _infer_raw(self, img_nchw: np.ndarray) -> np.ndarray:
        outputs = self.session.run(None, {self.input_name: img_nchw})
        return outputs[0]  # shape [1, 84, 8400]
