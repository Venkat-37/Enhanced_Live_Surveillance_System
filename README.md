#  Enhanced Live Surveillance System

Real-time object detection and monitoring system built with **YOLOv8** and **Streamlit**. Features multi-zone monitoring, intelligent alerts with cooldown, SQLite logging, snapshot capture, and a full analytics dashboard.

---

##  Architecture

```mermaid
graph TD
    subgraph Input Sources
        A[🎥 Webcam] --> D
        B[📡 RTSP/IP Camera] --> D
        C[🎬 Video File] --> D
    end

    subgraph Performance Layer
        D[Frame Manager<br/>Threading + Queue + Frame Skip]
    end

    subgraph ML Core
        D --> E[Detection Engine<br/>YOLOv8 · Multi-class · ROI Zones]
    end

    subgraph Event Handling
        E --> F[Alert Engine<br/>Sound · Email · SMS · Cooldown]
        E --> G[SQLite Logger<br/>Events + Snapshots]
        E --> H[Snapshot Saver<br/>Annotated Frames]
    end

    subgraph UI Layer
        F --> I[Streamlit Dashboard<br/>Live Feed · Analytics · ROI Config · Logs]
        G --> I
        H --> I
        I --> J[Config Sidebar<br/>Confidence · Cooldown · Camera · Alerts]
    end

    style D fill:#1a5276,color:#fff
    style E fill:#4a235a,color:#fff
    style F fill:#0e6655,color:#fff
    style G fill:#0e6655,color:#fff
    style H fill:#0e6655,color:#fff
    style I fill:#784212,color:#fff
    style J fill:#6e2c00,color:#fff
```

---

## Features

| Feature | Description |
|---------|-------------|
| **YOLOv8 Detection** | 10–20× faster than Faster R-CNN, multi-class support |
| **Threaded Capture** | Producer-consumer pattern with frame skipping |
| **Drawable ROI Zones** | Draw monitoring zones directly on the feed |
| **Per-Zone Cooldown** | Configurable alert cooldown per zone (no alarm flooding) |
| **SQLite Logging** | Structured event storage with analytics queries |
| **Snapshot Gallery** | Auto-saved annotated frames on detection |
| **Email / SMS Alerts** | SMTP + Twilio integration with env-based config |
| **RTSP Support** | Connect any ONVIF/RTSP compatible IP camera |
| **Plotly Analytics** | Detections/hour, zone distribution, class breakdown |
| **Config Sidebar** | Every parameter exposed — no hardcoded magic numbers |

---

##  Project Structure

```
surveillance_system/
├── app.py                  # Streamlit dashboard (main entry)
├── detection_engine.py     # YOLOv8 model wrapper
├── frame_manager.py        # Threaded frame capture + queue
├── alert_engine.py         # Sound, email, SMS + cooldown
├── db_logger.py            # SQLite detection logger
├── snapshot_saver.py       # Annotated frame saving
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── README.md               # This file
├── detections.db           # SQLite database (auto-created)
└── snapshots/              # Saved detection frames (auto-created)
```

---

##  Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure alerts (optional)

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

| Variable | Description |
|----------|-------------|
| `SMTP_HOST` | SMTP server (default: smtp.gmail.com) |
| `SMTP_PORT` | SMTP port (default: 587) |
| `SMTP_USER` | Your email address |
| `SMTP_PASS` | App-specific password |
| `ALERT_EMAIL_TO` | Alert recipient email |
| `TWILIO_SID` | Twilio Account SID |
| `TWILIO_AUTH_TOKEN` | Twilio Auth Token |
| `TWILIO_FROM` | Twilio phone number |
| `ALERT_SMS_TO` | SMS recipient number |

### 3. Run the application

```bash
streamlit run app.py
```

---

## 📷 Camera Support

| Source | How to Use |
|--------|-----------|
| **Webcam** | Select "Webcam" and choose device index (0 = default) |
| **RTSP/IP Camera** | Select "RTSP / IP Camera" and enter the stream URL |
| **Video File** | Select "Video File" and upload an MP4/AVI/MOV file |

> Any ONVIF/RTSP compatible IP camera works. Typical URL format: `rtsp://user:pass@192.168.1.100:554/stream`

---

##  Dashboard Tabs

1. ** Live Feed** — Real-time video with detection overlays and status cards (FPS, detections, alerts)
2. ** ROI Zones** — Draw monitoring rectangles with custom names. Detections inside zones trigger alerts.
3. ** Analytics** — Plotly charts: detections per hour, zone distribution pie, class breakdown bar
4. ** Snapshots** — Gallery of auto-saved annotated frames from detection events
5. ** Logs** — Searchable table of all detection records from SQLite

---

##  System Requirements

- Python 3.9+
- Webcam or IP camera (for live monitoring)
- ~500 MB disk (YOLOv8 nano model download on first run)

---

##  License

MIT
