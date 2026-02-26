"""
density.py — Crowd density classification.

Classifies the current crowd count into Low / Medium / High and
provides colour-coded overlays for the video feed.

Thresholds are configurable via utils.Config:
  - Low:    0 to Config.LOW_MAX (default 10)
  - Medium: Config.LOW_MAX+1 to Config.MEDIUM_MAX (default 25)
  - High:   > Config.MEDIUM_MAX
"""

import cv2
import numpy as np

from utils import Config, draw_text


class DensityEstimator:
    """Threshold-based crowd density classifier."""

    # --------------------------------------------------------------------- #
    #  Public API                                                            #
    # --------------------------------------------------------------------- #

    @staticmethod
    def classify(count: int) -> tuple[str, tuple]:
        """
        Classify *count* into a density level.

        Returns:
            (level_name, bgr_colour)
        """
        if count <= Config.LOW_MAX:
            return "Low", Config.COLOR_LOW
        elif count <= Config.MEDIUM_MAX:
            return "Medium", Config.COLOR_MEDIUM
        else:
            return "High", Config.COLOR_HIGH

    @staticmethod
    def draw_density_badge(frame: np.ndarray, count: int,
                           level: str, color: tuple) -> None:
        """
        Draw a coloured density badge in the top-right corner of *frame*.

        Shows both the people count and the density level with a
        semi-transparent background in the corresponding colour.
        """
        h, w = frame.shape[:2]

        # --- Badge background ---
        badge_w, badge_h = 280, 90
        x1 = w - badge_w - 15
        y1 = 15
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x1 + badge_w, y1 + badge_h), color, -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        # --- Border ---
        cv2.rectangle(frame, (x1, y1), (x1 + badge_w, y1 + badge_h), color, 2)

        # --- Text ---
        draw_text(frame, f"People: {count}", (x1 + 12, y1 + 30),
                  color=(255, 255, 255), scale=0.65, thickness=2)
        draw_text(frame, f"Density: {level}", (x1 + 12, y1 + 65),
                  color=(255, 255, 255), scale=0.65, thickness=2)
