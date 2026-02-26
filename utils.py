"""
utils.py — Configuration, logging, and helper utilities.

All tuneable thresholds and settings are centralized in the Config class.
Modify values there to customize detection sensitivity, density levels,
anomaly triggers, and visual appearance.
"""

import csv
import os
import time
from datetime import datetime

import cv2
import numpy as np


# =============================================================================
# CONFIGURATION — All tuneable thresholds and settings in one place
# =============================================================================


class Config:
    """Central configuration for the entire application."""

    # ---- Detection ----
    YOLO_MODEL = "yolov8n.pt"           # Ultralytics model name / path
    CONFIDENCE_THRESHOLD = 0.35         # Minimum confidence for person detection
    PERSON_CLASS_ID = 0                 # COCO class id for "person"

    # ---- Density thresholds ----
    LOW_MAX = 10                        # 0 – LOW_MAX  → Low density
    MEDIUM_MAX = 25                     # LOW_MAX+1 – MEDIUM_MAX → Medium density
    #                                    > MEDIUM_MAX → High density

    # ---- Density colours (BGR) ----
    COLOR_LOW = (0, 200, 0)             # Green
    COLOR_MEDIUM = (0, 220, 255)        # Yellow
    COLOR_HIGH = (0, 0, 255)            # Red

    # ---- Anomaly detection ----
    ANOMALY_SPIKE_PERCENT = 0.30        # 30 % spike triggers anomaly
    ANOMALY_WINDOW_SECONDS = 5          # Rolling window length (seconds)
    OVERCROWDING_THRESHOLD = 30         # Absolute count for overcrowding

    # ---- Logging ----
    CSV_FILE = "crowd_data.csv"
    ALERT_LOG_FILE = "alerts.log"
    GRAPH_FILE = "crowd_analysis.png"
    LOG_INTERVAL_SECONDS = 1.0          # How often to log a row

    # ---- Display ----
    WINDOW_NAME = "Crowd Monitoring System"
    BBOX_COLOR = (255, 180, 0)          # Bounding-box colour (BGR)
    ID_COLOR = (255, 255, 255)          # Track-ID text colour
    FONT_SCALE = 0.55
    FONT_THICKNESS = 2


# =============================================================================
# CSV LOGGER — stores (timestamp, count, density_level) per interval
# =============================================================================

class CSVLogger:
    """Append crowd-count rows to a CSV file at a configured interval."""

    def __init__(self, filepath: str = Config.CSV_FILE):
        self.filepath = filepath
        self._last_log_time = 0.0
        # Write header if file does not exist yet
        if not os.path.exists(self.filepath):
            with open(self.filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "crowd_count", "density_level"])

    def log(self, count: int, density_level: str, force: bool = False) -> None:
        """Log a row if enough time has elapsed (or *force* is True)."""
        now = time.time()
        if not force and (now - self._last_log_time) < Config.LOG_INTERVAL_SECONDS:
            return
        self._last_log_time = now
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.filepath, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, count, density_level])


# =============================================================================
# ALERT LOGGER — stores alert messages with timestamps
# =============================================================================

class AlertLogger:
    """Append alert messages to a plain-text log file."""

    def __init__(self, filepath: str = Config.ALERT_LOG_FILE):
        self.filepath = filepath

    def log(self, message: str) -> None:
        """Write an alert message to the log file and print to terminal."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}\n"
        with open(self.filepath, "a") as f:
            f.write(line)
        # Also print to terminal
        print(f"⚠  ALERT: {message}")


# =============================================================================
# FPS CALCULATOR
# =============================================================================

class FPSCounter:
    """Simple exponential-moving-average FPS counter."""

    def __init__(self, smoothing: float = 0.9):
        self._smoothing = smoothing
        self._prev_time = time.time()
        self._fps = 0.0

    def tick(self) -> float:
        """Call once per frame. Returns the smoothed FPS value."""
        now = time.time()
        dt = now - self._prev_time
        self._prev_time = now
        instant_fps = 1.0 / max(dt, 1e-6)
        self._fps = self._smoothing * self._fps + (1 - self._smoothing) * instant_fps
        return self._fps

    @property
    def fps(self) -> float:
        return self._fps


# =============================================================================
# DRAWING HELPERS
# =============================================================================


def draw_text(
    frame: np.ndarray,
    text: str,
    position: tuple,
    color: tuple = (255, 255, 255),
    scale: float = Config.FONT_SCALE,
    thickness: int = Config.FONT_THICKNESS,
    bg_color: tuple | None = None,
) -> None:
    """Draw text with optional background rectangle for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = position
    if bg_color is not None:
        cv2.rectangle(frame, (x, y - th - baseline), (x + tw, y + baseline), bg_color, -1)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def play_beep() -> None:
    """Play a short beep sound (best-effort, silent on failure)."""
    try:
        import winsound
        winsound.Beep(1000, 300)
    except Exception:
        # Not on Windows or no sound device — silently ignore
        pass
