"""
detection_engine.py — YOLOv8-based object detection wrapper.

Loads the YOLOv8-nano model once and exposes a simple `detect()` API that
returns structured detections with ROI zone filtering.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    """Single detection result."""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    in_zone: Optional[str] = None     # zone name if inside a monitored zone


@dataclass
class Zone:
    """A named rectangular ROI zone."""
    name: str
    x1: int
    y1: int
    x2: int
    y2: int
    color: Tuple[int, int, int] = (255, 0, 0)  # BGR


class DetectionEngine:
    """Wraps YOLOv8 for object detection with zone awareness."""

    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model = YOLO(model_path)

    # ── public API ───────────────────────────────────────────────

    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.5,
        target_classes: Optional[List[str]] = None,
        zones: Optional[List[Zone]] = None,
    ) -> List[Detection]:
        """
        Run YOLOv8 inference on *frame*.

        Parameters
        ----------
        frame : np.ndarray
            BGR image.
        confidence_threshold : float
            Minimum confidence to keep a detection.
        target_classes : list[str] | None
            If provided, only keep detections whose class name is in this list.
        zones : list[Zone] | None
            If provided, tag each detection with the zone it falls inside.

        Returns
        -------
        list[Detection]
        """
        results = self.model(frame, verbose=False)[0]
        detections: List[Detection] = []

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < confidence_threshold:
                continue

            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]

            if target_classes and cls_name not in target_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            zone_name = self._check_zones(x1, y1, x2, y2, zones)

            detections.append(
                Detection(
                    class_name=cls_name,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    in_zone=zone_name,
                )
            )

        return detections

    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        zones: Optional[List[Zone]] = None,
    ) -> np.ndarray:
        """Draw bounding boxes, labels, and zone overlays on a copy of *frame*."""
        annotated = frame.copy()

        # Draw zones first (so detections render on top)
        if zones:
            for z in zones:
                overlay = annotated.copy()
                cv2.rectangle(overlay, (z.x1, z.y1), (z.x2, z.y2), z.color, -1)
                cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0, annotated)
                cv2.rectangle(annotated, (z.x1, z.y1), (z.x2, z.y2), z.color, 2)
                cv2.putText(
                    annotated, z.name, (z.x1, z.y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, z.color, 2,
                )

        # Draw detections
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            color = (0, 0, 255) if d.in_zone else (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{d.class_name} {d.confidence:.0%}"
            cv2.putText(
                annotated, label, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2,
            )

            if d.in_zone:
                cv2.putText(
                    annotated, f"ALERT: {d.in_zone}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2,
                )

        return annotated

    # ── private helpers ──────────────────────────────────────────

    @staticmethod
    def _check_zones(
        x1: int, y1: int, x2: int, y2: int,
        zones: Optional[List[Zone]],
    ) -> Optional[str]:
        """Return the name of the first zone whose rectangle overlaps the bbox."""
        if not zones:
            return None
        for z in zones:
            if x1 < z.x2 and x2 > z.x1 and y1 < z.y2 and y2 > z.y1:
                return z.name
        return None
