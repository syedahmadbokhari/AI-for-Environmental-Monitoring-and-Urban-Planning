# 🧠 AI-Powered Environment Monitoring System

An AI-based real-time monitoring system designed to detect illegal dumping using CCTV video streams. The system combines computer vision, deep learning, and a modular backend to generate alerts and provide actionable insights through a dashboard.

---

## 🚀 Overview

Illegal dumping is a growing urban issue causing environmental damage and increased operational costs.

This project presents a real-time AI-powered monitoring system that transforms CCTV into an intelligent detection tool.

By combining computer vision and deep learning, the system automatically detects illegal dumping events, generates alerts, and provides actionable insights through a dashboard.

---

## ⚙️ Key Features

* 🎥 Real-time CCTV video processing
* 🧠 Motion detection and object tracking
* 🗑️ Waste classification using MobileNetV3
* ⚡ Event detection and alert generation
* 🌐 Backend API with Flask
* 📊 Interactive dashboard (Streamlit)
* 🗺️ GIS-based visualisation

---

## 🏗️ System Pipeline

1. Video input from CCTV
2. Motion detection filters frames
3. Object tracking monitors behaviour
4. Stationary objects are classified
5. Waste detection triggers event
6. Event sent to backend
7. Dashboard updates in real time

---

# 🏗️ System Architecture

![Architecture](diagrams/System%20Architecture%20v2.png)

---

## 🧩 Project Architecture Diagram

![Project Architecture](diagrams/Project%20Architecture%20Diagram.png)

---

## 🛠️ Tech Stack

* **Programming:** Python
* **Computer Vision:** OpenCV
* **AI Framework:** PyTorch + torchvision (MobileNetV3)
* **Backend:** Flask (REST API, MJPEG streaming, SSE)
* **Version Control:** Git & GitHub

---

## ▶️ Installation

```bash
git clone https://github.com/tsabesoo/ai-for-environmental-monitoring-and-urban-planning.git
cd ai-for-environmental-monitoring-and-urban-planning
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the standalone CLI demo (single video file):

```bash
python main.py --video videos/video.mp4 --model models/best_model.pth
```

Run the Flask backend (multi-camera dashboard):

```bash
python dashboard/app.py --model models/best_model.pth
```

Then open `http://localhost:5000` in your browser.

---

## 📁 Project Structure

```
ai-for-environmental-monitoring-and-urban-planning/
│
├── core/                    # Core detection modules
│   ├── __init__.py
│   ├── classifier.py        # MobileNetV3 inference
│   └── pipeline.py          # 4-stage detection pipeline + CameraPipeline
│
├── dashboard/               # Flask backend + frontend
│   └── app.py               # REST API + MJPEG streams + SSE events
│
├── data/                    # Dataset (raw + processed)
├── diagrams/                # System architecture diagrams
├── docs/                    # Design, evaluation, ethics, PM documents
├── logs/                    # Runtime event logs
├── models/                  # Trained model checkpoints (MobileNetV3)
├── training/                # Model training scripts (train.ipynb)
├── videos/                  # Input video files (add your own here)
│
├── archive/                 # Prototype versions v0–v7 (historical reference)
│
├── config.yaml              # Central configuration (all tunable parameters)
├── main.py                  # CLI demo entry point (single video)
├── requirements.txt
└── README.md
```

---

## 🔄 Version Evolution

| Version | Description |
|---------|-------------|
| v0 | Initial concept and basic pipeline setup |
| v1 | Motion detection implemented |
| v2 | Object tracking introduced |
| v3 | Dashboard integration |
| v4 | Backend system (Flask API) developed |
| v5 | Event detection and alert system refined |
| v6 | Optimised real-time system |
| v7 | Dataset tooling, YOLO experiments, methodology report |

---

## ⚖️ Ethical Considerations

* No facial recognition or identity tracking
* Data minimisation: only relevant event data stored
* Transparent event logging for accountability
* Designed in compliance with GDPR principles
* Ethical risks evaluated using ALTAI framework

---

## 🌍 Impact

* Enables real-time monitoring of illegal dumping
* Reduces manual CCTV workload
* Supports faster response to incidents
* Contributes to cleaner and safer urban environments
* Aligns with smart city initiatives

---

## 🔮 Future Improvements

* Integration of advanced models (e.g., YOLO)
* Cloud deployment for scalability
* Edge AI optimisation
* Improved dataset diversity

---

## 👨‍💻 Authors

Syed Bokhari, Sabal Nemkul, Thatsara Abesooriya

---

## 📄 License

This project is developed for educational and research purposes.
(Add your license here, e.g., MIT License)

