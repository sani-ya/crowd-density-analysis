"""
tracking.py — Multi-object tracking with DeepSORT.

Wraps the `deep_sort_realtime` library to assign persistent IDs
to detected persons across frames.
"""

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

from utils import Config, draw_text


class PersonTracker:
    """Maintains a DeepSORT tracker for person detections."""

    def __init__(self, max_age: int = 30, n_init: int = 3,
                 max_iou_distance: float = 0.7):
        """
        Args:
            max_age:           Frames to keep a lost track before deletion.
            n_init:            Hits before a track is confirmed.
            max_iou_distance:  Maximum IoU distance for matching.
        """
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_iou_distance=max_iou_distance,
        )

    # --------------------------------------------------------------------- #
    #  Public API                                                            #
    # --------------------------------------------------------------------- #

    def update(self, frame: np.ndarray,
               detections: list[list[float]]) -> list[tuple]:
        """
        Feed new detections into the tracker and return confirmed tracks.

        Args:
            frame:       Current BGR frame.
            detections:  List of [x1, y1, x2, y2, confidence].

        Returns:
            List of (track_id, x1, y1, x2, y2) for each confirmed track.
        """
        # deep-sort-realtime expects detections as [(bbox, conf, class), ...]
        # where bbox = [x, y, w, h]
        ds_detections = []
        for det in detections:
            x1, y1, x2, y2, conf = det
            w = x2 - x1
            h = y2 - y1
            ds_detections.append(([x1, y1, w, h], conf, "person"))

        tracks = self.tracker.update_tracks(ds_detections, frame=frame)

        results: list[tuple] = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()  # [left, top, right, bottom]
            x1, y1, x2, y2 = [int(v) for v in ltrb]
            results.append((track_id, x1, y1, x2, y2))

        return results

    @staticmethod
    def draw_tracks(frame: np.ndarray,
                    tracks: list[tuple]) -> None:
        """Draw track IDs above each bounding box onto *frame* in-place."""
        for track_id, x1, y1, x2, y2 in tracks:
            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), Config.BBOX_COLOR, 2)
            # Track ID label
            label = f"ID {track_id}"
            draw_text(frame, label, (x1, y1 - 10), color=Config.ID_COLOR,
                      scale=0.5, thickness=1, bg_color=(50, 50, 50))
