# Real-Time Crowd Density Estimation & Anomaly Detection

> **YOLOv8 + DeepSORT** smart surveillance system for real-time person detection, tracking, density estimation, anomaly alerting, and data logging.

---

## 📋 Project Description

This application processes live video (webcam or file) to:

| Feature | Details |
|---|---|
| **Person Detection** | YOLOv8n pre-trained on COCO — filters for `person` class only |
| **Multi-Object Tracking** | DeepSORT assigns and maintains unique IDs across frames |
| **Density Classification** | Colour-coded **Low / Medium / High** based on configurable thresholds |
| **Anomaly Detection** | Alerts on sudden crowd spikes (>30 %) and overcrowding |
| **Data Logging** | Timestamped CSV (`crowd_data.csv`) + alert log (`alerts.log`) |
| **Visualization** | Matplotlib graph (`crowd_analysis.png`) generated on exit |
| **GUI** | Optional Tkinter control panel with Start / Stop buttons |
| **ROI Selection** | Optionally limit detection to a drawn region of interest |

---

## 🛠️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/crowd-density-analysis.git
cd crowd-density-analysis

# 2. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note:** The YOLOv8n model weights are downloaded automatically on first run (~6 MB).

---

## 🚀 How to Run

```bash
# Webcam (default)
python main.py

# Video file
python main.py --source path/to/video.mp4

# With ROI selection
python main.py --roi

# Tkinter GUI
python main.py --gui
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--source` | `0` | Webcam index or path to video file |
| `--roi` | off | Draw a region of interest on the first frame |
| `--gui` | off | Launch the graphical control panel |

Press **`q`** in the OpenCV window to quit.

---

## 📂 Project Structure

```
crowd-density-analysis/
├── main.py            # Entry point — CLI / GUI / video loop
├── detection.py       # YOLOv8 person detector
├── tracking.py        # DeepSORT multi-object tracker
├── density.py         # Threshold-based density classifier
├── anomaly.py         # Spike & overcrowding detector
├── utils.py           # Config, loggers, FPS counter, drawing helpers
├── requirements.txt   # Python dependencies
├── README.md          # This file
├── crowd_data.csv     # (generated) timestamped crowd counts
├── alerts.log         # (generated) anomaly alert log
└── crowd_analysis.png # (generated) crowd-count-over-time graph
```

---

## 📊 Expected Output

1. **Live Video Window** — bounding boxes with confidence scores, unique track IDs, colour-coded density badge, FPS counter, and red warning overlay on anomalies.
2. **Terminal** — real-time anomaly alerts printed with timestamps.
3. **crowd_data.csv** — one row per second with `timestamp, crowd_count, density_level`.
4. **alerts.log** — timestamped anomaly messages.
5. **crowd_analysis.png** — line chart of crowd count over time with density-level shading.

### Sample Screenshots Description

| View | Description |
|---|---|
| Normal monitoring | Green "Low Density" badge, bounding boxes with IDs, FPS counter |
| Medium crowd | Yellow "Medium Density" badge, 15+ tracked persons |
| Anomaly detected | Red warning banner across screen, "SPIKE DETECTED" message |
| Analysis graph | Line chart with green / orange / red shading for density zones |

---

## ⚙️ Configuration

All thresholds are centralized in `utils.py → Config`:

```python
LOW_MAX = 10              # 0-10 people → Low
MEDIUM_MAX = 25           # 11-25 → Medium; >25 → High
ANOMALY_SPIKE_PERCENT = 0.30
OVERCROWDING_THRESHOLD = 30
CONFIDENCE_THRESHOLD = 0.35
```

---

## 🔮 Future Improvements

- **Heatmap overlay** — visualize crowd distribution spatially.
- **Zone-based counting** — define multiple ROIs with independent thresholds.
- **Database logging** — store data in SQLite / PostgreSQL for long-term analysis.
- **Web dashboard** — Flask / Streamlit real-time dashboard with live charts.
- **Email / SMS alerts** — integrate Twilio or SMTP for remote notifications.
- **GPU acceleration** — TensorRT export for higher FPS on edge devices.
- **Cloud deployment** — run on AWS / Azure with RTSP stream input.
- **Re-identification** — cross-camera person re-ID using appearance features.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
