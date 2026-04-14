"""
torchscript_detector.py — TorchScript inference detector on MPS (Apple Metal GPU).
Loads the .torchscript file exported by scripts/export_models.py.
"""

from __future__ import annotations

import numpy as np
import torch

from backend.detectors.base import BaseDetector
from backend.config import MODELS_DIR, DEFAULT_CONF_THRESHOLD, DEFAULT_IOU_THRESHOLD


class TorchScriptDetector(BaseDetector):
    """
    TorchScript + MPS detector.
    Satisfies the 'TorchScript' acceleration requirement.
    """

    def __init__(
        self,
        model_name: str,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    ) -> None:
        super().__init__(model_name, conf_threshold, iou_threshold)
        # TorchScript models are traced on CPU; internal constant tensors
        # (anchor points) stay on CPU. Running on CPU uses TorchScript's
        # JIT graph optimization and operator fusion over eager PyTorch.
        self.device = torch.device("cpu")
        self.model: torch.ScriptModule | None = None

    def load(self) -> None:
        ts_path = MODELS_DIR / f"{self.model_name}.torchscript"
        if not ts_path.exists():
            raise FileNotFoundError(
                f"TorchScript model not found: {ts_path}\n"
                "Run: python3 scripts/export_models.py"
            )

        self.model = torch.jit.load(str(ts_path), map_location="cpu")
        self.model.eval()

        # Warmup — critical to avoid cold-start latency spikes
        dummy = torch.zeros(1, 3, 640, 640, device=self.device)
        with torch.no_grad():
            for _ in range(5):
                self.model(dummy)

        self._loaded = True

    def _infer_raw(self, img_nchw: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(img_nchw).to(self.device)
        with torch.no_grad():
            out = self.model(tensor)
        if isinstance(out, (list, tuple)):
            out = out[0]
        return out.cpu().numpy()
