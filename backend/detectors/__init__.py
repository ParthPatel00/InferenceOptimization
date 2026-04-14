from backend.types import Detection, InferenceResult
from .base import BaseDetector
from .pytorch_detector import PyTorchDetector
from .torchscript_detector import TorchScriptDetector
from .onnx_detector import ONNXCoreMLDetector

__all__ = [
    "BaseDetector",
    "Detection",
    "InferenceResult",
    "PyTorchDetector",
    "TorchScriptDetector",
    "ONNXCoreMLDetector",
]
