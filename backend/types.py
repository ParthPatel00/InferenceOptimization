"""
types.py — Shared dataclasses used across detectors and utils.
Kept in a standalone module to avoid circular imports.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Detection:
    bbox: list[float]      # [x1, y1, x2, y2] in original pixel coordinates
    class_id: int
    class_name: str
    confidence: float


@dataclass
class InferenceResult:
    detections: list[Detection]
    latency_ms: float
    image_width: int
    image_height: int
