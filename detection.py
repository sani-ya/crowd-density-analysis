"""
detection.py — Person detection using YOLOv8 (Ultralytics).

Only the "person" class (COCO id 0) is kept.  Each detection is returned
as [x1, y1, x2, y2, confidence] in pixel coordinates.
"""

import cv2
import numpy as np
from ultralytics import YOLO

from utils import Config, draw_text


class PersonDetector:
    """Wraps a YOLOv8 model and filters for the 'person' class."""

    def __init__(self, model_path: str = Config.YOLO_MODEL,
                 confidence: float = Config.CONFIDENCE_THRESHOLD):
        """
        Args:
            model_path:  Name or path of the YOLO model (downloaded on first use).
            confidence:  Minimum confidence threshold.
        """
        self.model = YOLO(model_path)
        self.confidence = confidence

    # --------------------------------------------------------------------- #
    #  Public API                                                            #
    # --------------------------------------------------------------------- #

    def detect(self, frame: np.ndarray) -> list[list[float]]:
        """
        Run inference on *frame* and return person detections.

        Returns:
            List of [x1, y1, x2, y2, confidence] for every detected person.
        """
        results = self.model(frame, verbose=False, conf=self.confidence)
        detections: list[list[float]] = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id != Config.PERSON_CLASS_ID:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append([x1, y1, x2, y2, conf])

        return detections

    @staticmethod
    def draw_detections(frame: np.ndarray,
                        detections: list[list[float]]) -> None:
        """Draw bounding boxes and confidence scores onto *frame* in-place."""
        for det in detections:
            x1, y1, x2, y2, conf = det
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), Config.BBOX_COLOR, 2)
            # Confidence label
            label = f"{conf:.0%}"
            draw_text(frame, label, (x1, y1 - 5), color=Config.BBOX_COLOR,
                      scale=0.45, thickness=1)
