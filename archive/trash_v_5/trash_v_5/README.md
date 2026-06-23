# Illegal Dumping Detection System

AI-powered CCTV monitoring system that automatically detects illegal trash dumping events in real time using computer vision and deep learning.

> **University of Bradford** — Final Year Project

---

## Overview

The system processes live or recorded camera feeds through a 4-stage pipeline:

| Stage | What it does |
|-------|-------------|
| **1 — Background Subtraction** | MOG2 algorithm isolates moving foreground objects |
| **2 — Tracking** | Centroid-based tracker follows each detected blob across frames |
| **3 — Event Detection** | Flags tracks that become stationary (object left behind) |
| **4 — Classifier Confirmation** | MobileNetV3-Large CNN confirms the blob is actually trash |

Confirmed events trigger a real-time alert in the web dashboard, save a snapshot, and append a record to an event log.

---

## Project Structure

```
.
├── app.py                  # Flask web server (MJPEG streams, SSE events, REST API)
├── main.py                 # Standalone CLI demo (no Flask)
├── core/
│   ├── __init__.py
│   ├── classifier.py       # MobileNetV3 inference module + CLI
│   └── pipeline.py         # Full 4-stage CV pipeline + CameraPipeline thread
├── models/
│   └── best_model.pth      # Trained MobileNetV3-Large checkpoint
├── static/
│   └── index.html          # Single-file web dashboard
├── videos/                 # Test video files
├── logs/
│   └── events.jsonl        # Append-only event log (auto-created)
├── snapshots/              # Saved JPEG frames for each event (auto-created)
└── requirements.txt
```

---

## Requirements

- Python 3.9+
- CUDA-capable GPU (optional — CPU fallback works)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Web Dashboard

```bash
python app.py --model models/best_model.pth --threshold 0.5
```

Then open **http://localhost:5000** in your browser.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `models/best_model.pth` | Path to trained checkpoint |
| `--threshold` | `0.5` | Classifier confidence cutoff |
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `5000` | Bind port |
| `--debug` | off | Flask debug mode |

---

## Running the CLI Demo

Single-camera processing without the web server:

```bash
python main.py --video videos/video.mp4 --model models/best_model.pth --threshold 0.5
```

### Single-image classifier test

```bash
python -m core.classifier --image test.jpg --model models/best_model.pth
```

---

## Web Dashboard

### Camera Grid
- Supports up to **4 simultaneous cameras**
- Accepts **local video files** (`.mp4`, etc.) or **RTSP streams** (`rtsp://…`) or **webcam index** (`0`)
- Live MJPEG video feed with bounding-box annotations
- Per-camera start / stop / remove controls

### Event Log (sidebar)
- Real-time alerts via **Server-Sent Events (SSE)**
- Snapshot thumbnail for each confirmed trash event
- Click any event to view the full snapshot in a modal
- Events persist across page reloads (loaded from `logs/events.jsonl`)

### Settings Panel (⚙ button, top-right)
Configure the pipeline at runtime — no restart needed for most values:

| Setting | Hot-apply? | Description |
|---------|-----------|-------------|
| Confidence Threshold | ✅ immediately | Minimum classifier score to confirm trash |
| JPEG Quality | ✅ immediately | Stream encoding quality (10–100) |
| Min Blob Area | ✅ next frame | Minimum foreground contour area (px²) |
| Max Tracking Distance | ✅ next frame | Max centroid movement to link to existing track |
| Max Missed Frames | ✅ next frame | Frames before a lost track is deleted |
| Stationary Distance | ✅ next frame | Movement threshold to count as stationary |
| Stationary Min Frames | ✅ next frame | Frames stationary before event is triggered |
| BG History | ❌ new cameras only | MOG2 background model history length |
| BG Var Threshold | ❌ new cameras only | MOG2 foreground sensitivity |

---

## Bounding Box Colors

| Color | Meaning |
|-------|---------|
| 🟣 Magenta | Confirmed **trash** (classifier said yes) |
| 🟡 Yellow | Confirmed **not trash** (classifier said no) |
| 🔴 Red | Stationary event (awaiting/failed classifier) |
| 🔵 Cyan | Track stationary but event not yet fired |
| 🟢 Green | Active moving track |

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Dashboard UI |
| GET | `/api/cameras` | List all cameras |
| POST | `/api/cameras` | Add camera `{name, source}` |
| DELETE | `/api/cameras/<id>` | Remove camera |
| POST | `/api/cameras/<id>/start` | Start pipeline |
| POST | `/api/cameras/<id>/stop` | Stop pipeline |
| GET | `/video/<id>` | MJPEG stream |
| GET | `/events` | SSE event stream |
| GET | `/api/events?n=100` | Last N logged events (JSON) |
| GET | `/snapshots/<file>` | Serve snapshot image |
| GET | `/api/settings` | Get current pipeline settings |
| POST | `/api/settings` | Update pipeline settings |

---

## Model

The classifier is a **MobileNetV3-Large** pretrained on ImageNet and fine-tuned on a custom trash/no-trash dataset. Inference runs at the point of event detection (Stage 4), not on every frame.

To retrain or use a different checkpoint, pass its path via `--model`.
