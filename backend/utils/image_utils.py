"""
image_utils.py — Shared preprocessing and postprocessing for YOLO models.

All YOLO models (YOLOv8, YOLO11) share the same input/output format:
  Input : float32 NCHW [1, 3, 640, 640], values in [0, 1], RGB
  Output: float32      [1, 84, 8400], pre-NMS
            axis-1 layout: [cx, cy, w, h, class_0_score, ..., class_79_score]
"""

from __future__ import annotations

import numpy as np
import cv2

from backend.config import COCO_CLASSES, INPUT_SIZE
from backend.types import Detection


def preprocess_image(
    image_bgr: np.ndarray,
    target_size: int = INPUT_SIZE,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """
    Letterbox-resize → RGB → normalize → NCHW.

    Returns:
        img_nchw : float32 array shape [1, 3, target_size, target_size]
        scale    : float, scale factor applied to the shorter side
        pad      : (pad_w, pad_h) pixels added on each side
    """
    orig_h, orig_w = image_bgr.shape[:2]

    # Scale to fit inside target_size × target_size
    scale = min(target_size / orig_w, target_size / orig_h)
    new_w = round(orig_w * scale)
    new_h = round(orig_h * scale)

    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Letterbox padding
    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2

    padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    padded[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

    # BGR → RGB, HWC → NCHW, /255
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    nchw = rgb.transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0

    return nchw, scale, (pad_w, pad_h)


def postprocess_detections(
    raw_output: np.ndarray,
    scale: float,
    pad: tuple[int, int],
    orig_shape: tuple[int, int],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> list[Detection]:
    """
    Decode raw YOLO output [1, 84, 8400] → list of Detection objects.

    Steps:
      1. Transpose to [8400, 84]
      2. Split into box [cx, cy, w, h] and class scores [80]
      3. Filter rows where max class score > conf_threshold
      4. Convert to xyxy in original image coords
      5. Apply NMS (numpy-based, no torch dependency)
    """
    orig_h, orig_w = orig_shape
    pad_w, pad_h = pad

    # [1, 84, 8400] → [8400, 84]
    preds = raw_output[0].T  # shape [8400, 84]

    box_raw = preds[:, :4]         # cx, cy, w, h (in letterboxed 640x640 space)
    class_scores = preds[:, 4:]    # [8400, 80]

    class_ids = class_scores.argmax(axis=1)
    confidences = class_scores.max(axis=1)

    # Filter by confidence
    mask = confidences > conf_threshold
    if not mask.any():
        return []

    box_raw = box_raw[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    # Convert cxcywh → xyxy (still in 640x640 letterboxed space)
    x1 = box_raw[:, 0] - box_raw[:, 2] / 2
    y1 = box_raw[:, 1] - box_raw[:, 3] / 2
    x2 = box_raw[:, 0] + box_raw[:, 2] / 2
    y2 = box_raw[:, 1] + box_raw[:, 3] / 2

    # Remove letterbox padding and scale back to original image coords
    x1 = (x1 - pad_w) / scale
    y1 = (y1 - pad_h) / scale
    x2 = (x2 - pad_w) / scale
    y2 = (y2 - pad_h) / scale

    # Clip to image bounds
    x1 = np.clip(x1, 0, orig_w)
    y1 = np.clip(y1, 0, orig_h)
    x2 = np.clip(x2, 0, orig_w)
    y2 = np.clip(y2, 0, orig_h)

    boxes = np.stack([x1, y1, x2, y2], axis=1)

    # Per-class NMS
    keep_indices = _nms_numpy(boxes, confidences, class_ids, iou_threshold)

    detections: list[Detection] = []
    for idx in keep_indices:
        cid = int(class_ids[idx])
        detections.append(
            Detection(
                bbox=[float(x1[idx]), float(y1[idx]), float(x2[idx]), float(y2[idx])],
                class_id=cid,
                class_name=COCO_CLASSES.get(cid, f"class_{cid}"),
                confidence=float(confidences[idx]),
            )
        )

    return detections


def _nms_numpy(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
) -> list[int]:
    """Greedy NMS per class. Returns list of kept indices into the input arrays."""
    keep: list[int] = []
    for cls in np.unique(class_ids):
        cls_mask = class_ids == cls
        cls_indices = np.where(cls_mask)[0]
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]

        order = cls_scores.argsort()[::-1]
        kept_local: list[int] = []

        while len(order) > 0:
            best = order[0]
            kept_local.append(best)
            if len(order) == 1:
                break

            rest = order[1:]
            ious = _iou(cls_boxes[best], cls_boxes[rest])
            order = rest[ious <= iou_threshold]

        keep.extend(cls_indices[i] for i in kept_local)

    return keep


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU between one box [4] and many boxes [N, 4]."""
    inter_x1 = np.maximum(box[0], boxes[:, 0])
    inter_y1 = np.maximum(box[1], boxes[:, 1])
    inter_x2 = np.minimum(box[2], boxes[:, 2])
    inter_y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = area_box + area_boxes - inter_area

    return inter_area / (union_area + 1e-6)
