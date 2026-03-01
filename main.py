"""
main.py — Entry point for the Real-Time Crowd Monitoring System.

Usage
-----
    python main.py                      # webcam (default)
    python main.py --source video.mp4   # video file
    python main.py --gui                # launch Tkinter GUI
    python main.py --roi                # enable ROI selection
"""

import argparse
import sys
import threading

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be before pyplot import
import matplotlib.pyplot as plt  # noqa: E402

from detection import PersonDetector  # noqa: E402
from tracking import PersonTracker  # noqa: E402
from density import DensityEstimator  # noqa: E402
from anomaly import AnomalyDetector  # noqa: E402
from utils import Config, CSVLogger, FPSCounter, draw_text  # noqa: E402


# ============================================================================
# GRAPH GENERATION
# ============================================================================

def generate_graph(csv_path: str = Config.CSV_FILE,
                   out_path: str = Config.GRAPH_FILE) -> None:
    """Read the CSV log and produce a crowd-count vs time line chart."""
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            print("[INFO] No data to plot.")
            return
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df["timestamp"], df["crowd_count"],
                color="#3b82f6", linewidth=1.5, marker="o", markersize=3)
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Crowd Count", fontsize=12)
        ax.set_title("Crowd Count Over Time", fontsize=14, fontweight="bold")

        # Density-level shading
        ax.axhspan(0, Config.LOW_MAX, alpha=0.08, color="green", label="Low")
        ax.axhspan(Config.LOW_MAX, Config.MEDIUM_MAX, alpha=0.08,
                   color="orange", label="Medium")
        ax.axhspan(Config.MEDIUM_MAX, df["crowd_count"].max() + 5,
                   alpha=0.08, color="red", label="High")
        ax.legend(loc="upper left")

        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[INFO] Graph saved to {out_path}")
    except Exception as e:
        print(f"[WARN] Could not generate graph: {e}")


# ============================================================================
# ROI SELECTION
# ============================================================================

def select_roi(frame: np.ndarray) -> tuple | None:
    """Let the user draw a rectangle ROI on the first frame."""
    roi = cv2.selectROI("Select ROI — press ENTER to confirm, C to cancel",
                        frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI — press ENTER to confirm, C to cancel")
    if roi == (0, 0, 0, 0):
        return None
    return roi  # (x, y, w, h)


def crop_to_roi(frame: np.ndarray, roi: tuple) -> np.ndarray:
    """Crop *frame* to the selected ROI."""
    x, y, w, h = roi
    return frame[y:y + h, x:x + w]


# ============================================================================
# MAIN VIDEO PROCESSING LOOP
# ============================================================================

class CrowdMonitor:
    """Encapsulates the full processing pipeline.

    Pipeline per frame:
        read → (optional ROI crop) → detect → track → density →
        anomaly → draw overlays → log → display
    """

    def __init__(self, source, use_roi: bool = False):
        self.source = source
        self.use_roi = use_roi
        self.running = False
        self._roi: tuple | None = None

        # Pipeline components
        self.detector = PersonDetector()
        self.tracker = PersonTracker()
        self.density_estimator = DensityEstimator()
        self.anomaly_detector = AnomalyDetector()
        self.csv_logger = CSVLogger()
        self.fps_counter = FPSCounter()

    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Open the video source and run the processing loop."""
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video source: {self.source}")
            print("[HINT]  If using a webcam, make sure it is connected.")
            print("[HINT]  If using a file, check that the path exists.")
            sys.exit(1)

        self.running = True
        print(f"[INFO] Started monitoring (source={self.source}). Press 'q' to quit.")

        # ROI selection on first frame
        if self.use_roi:
            ret, first_frame = cap.read()
            if ret:
                self._roi = select_roi(first_frame)
                if self._roi:
                    print(f"[INFO] ROI selected: {self._roi}")
                else:
                    print("[INFO] No ROI selected — using full frame.")
            # Seek back to start
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        try:
            self._loop(cap)
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user.")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.running = False
            print("[INFO] Generating analysis graph …")
            generate_graph()
            print("[INFO] Done.")

    # ------------------------------------------------------------------ #

    def stop(self) -> None:
        """Signal the processing loop to stop (used by the GUI)."""
        self.running = False

    # ------------------------------------------------------------------ #

    def _loop(self, cap: cv2.VideoCapture) -> None:
        """Core frame-by-frame processing loop."""
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of video stream.")
                break

            # Optional ROI crop
            process_frame = crop_to_roi(frame, self._roi) if self._roi else frame

            # 1 — Detect persons
            detections = self.detector.detect(process_frame)

            # 2 — Track with DeepSORT
            tracks = self.tracker.update(process_frame, detections)

            # 3 — Classify density
            count = len(tracks)
            level, color = self.density_estimator.classify(count)

            # 4 — Check for anomalies
            anomalies = self.anomaly_detector.check(count)

            # 5 — Draw everything on the frame
            self.tracker.draw_tracks(process_frame, tracks)
            self.density_estimator.draw_density_badge(process_frame, count, level, color)
            self.anomaly_detector.draw_warning(process_frame, anomalies)

            # FPS overlay
            fps = self.fps_counter.tick()
            draw_text(process_frame, f"FPS: {fps:.1f}", (15, 30),
                      color=(0, 255, 0), scale=0.65, thickness=2)

            # If ROI was used, paste back into the original frame for display
            if self._roi:
                x, y, w, h = self._roi
                frame[y:y + h, x:x + w] = process_frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
                display = frame
            else:
                display = process_frame

            # 6 — Log data
            self.csv_logger.log(count, level)

            # 7 — Show
            cv2.imshow(Config.WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[INFO] Quit requested.")
                break


# ============================================================================
# TKINTER GUI (optional)
# ============================================================================

def launch_gui(source) -> None:
    """Simple Tkinter GUI with Start / Stop buttons."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        print("[ERROR] Tkinter is not available. Run without --gui.")
        sys.exit(1)

    monitor: CrowdMonitor | None = None
    monitor_thread: threading.Thread | None = None

    # ---- window ----
    root = tk.Tk()
    root.title("Crowd Monitoring System")
    root.geometry("420x280")
    root.resizable(False, False)
    root.configure(bg="#1e1e2e")

    title_lbl = tk.Label(root, text="🎥  Crowd Monitor",
                         font=("Segoe UI", 16, "bold"),
                         bg="#1e1e2e", fg="#cdd6f4")
    title_lbl.pack(pady=(18, 8))

    # Source entry
    src_frame = tk.Frame(root, bg="#1e1e2e")
    src_frame.pack(pady=4)
    tk.Label(src_frame, text="Source:", bg="#1e1e2e", fg="#bac2de",
             font=("Segoe UI", 10)).pack(side="left", padx=4)
    src_var = tk.StringVar(value=str(source))
    src_entry = tk.Entry(src_frame, textvariable=src_var, width=25,
                         font=("Consolas", 10))
    src_entry.pack(side="left", padx=4)

    def browse_file():
        path = filedialog.askopenfilename(
            filetypes=[("Video", "*.mp4 *.avi *.mkv *.mov"), ("All", "*.*")])
        if path:
            src_var.set(path)

    tk.Button(src_frame, text="📂", command=browse_file,
              font=("Segoe UI", 10)).pack(side="left")

    # ROI checkbox
    roi_var = tk.BooleanVar(value=False)
    tk.Checkbutton(root, text="Enable ROI selection", variable=roi_var,
                   bg="#1e1e2e", fg="#a6adc8", selectcolor="#313244",
                   activebackground="#1e1e2e", activeforeground="#cdd6f4",
                   font=("Segoe UI", 10)).pack(pady=4)

    status_var = tk.StringVar(value="Ready")
    status_lbl = tk.Label(root, textvariable=status_var, bg="#1e1e2e",
                          fg="#a6e3a1", font=("Segoe UI", 10, "italic"))
    status_lbl.pack(pady=4)

    def on_start():
        nonlocal monitor, monitor_thread
        if monitor_thread and monitor_thread.is_alive():
            return

        src = src_var.get()
        try:
            src = int(src)
        except ValueError:
            pass

        try:
            status_var.set("Initializing …")
            root.update_idletasks()
            monitor = CrowdMonitor(source=src, use_roi=roi_var.get())
            monitor_thread = threading.Thread(target=monitor.start, daemon=True)
            monitor_thread.start()
            status_var.set("Monitoring …")
            start_btn.config(state="disabled")
            stop_btn.config(state="normal")
            # Periodically check if thread died
            def check_thread():
                if not monitor_thread.is_alive():
                    status_var.set("Stopped / Finished.")
                    start_btn.config(state="normal")
                    stop_btn.config(state="disabled")
                else:
                    root.after(500, check_thread)
            check_thread()
        except Exception as e:
            from tkinter import messagebox
            messagebox.showerror("Error", f"Failed to start monitoring:\n{e}")
            status_var.set("Error on start.")

    def on_stop():
        if monitor:
            monitor.stop()
            status_var.set("Stopping …")

    btn_frame = tk.Frame(root, bg="#1e1e2e")
    btn_frame.pack(pady=12)
    start_btn = tk.Button(btn_frame, text="▶  Start", command=on_start, width=12,
                          bg="#a6e3a1", fg="#1e1e2e", font=("Segoe UI", 11, "bold"),
                          relief="flat")
    start_btn.pack(side="left", padx=8)
    stop_btn = tk.Button(btn_frame, text="⏹  Stop", command=on_stop, width=12,
                         bg="#f38ba8", fg="#1e1e2e", font=("Segoe UI", 11, "bold"),
                         relief="flat", state="disabled")
    stop_btn.pack(side="left", padx=8)

    root.protocol("WM_DELETE_WINDOW", lambda: (on_stop(), root.destroy()))
    root.mainloop()


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-Time Crowd Density Estimation & Anomaly Detection")
    parser.add_argument("--source", default=0,
                        help="Video source: webcam index (default 0) or file path")
    parser.add_argument("--roi", action="store_true",
                        help="Enable Region of Interest (ROI) selection")
    parser.add_argument("--gui", action="store_true",
                        help="Launch the Tkinter GUI instead of direct monitoring")
    return parser.parse_args()


def main() -> None:
    """Application entry point."""
    args = parse_args()
    source = args.source
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass

    if args.gui:
        launch_gui(source)
    else:
        monitor = CrowdMonitor(source=source, use_roi=args.roi)
        monitor.start()


if __name__ == "__main__":
    main()
