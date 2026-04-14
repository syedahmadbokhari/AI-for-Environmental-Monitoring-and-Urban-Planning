# 🧠 AI-Powered Illegal Dumping Detection System

An AI-based real-time monitoring system designed to detect illegal dumping using CCTV video streams. The system combines computer vision, deep learning, and a modular backend to generate alerts and provide actionable insights through a dashboard.

---

## 🚀 Overview

Illegal dumping is a growing urban issue that leads to environmental damage and increased operational costs. This project presents a real-time AI solution that automates detection using CCTV feeds, reducing reliance on manual monitoring.

---

## ⚙️ Key Features

- 🎥 Real-time CCTV video processing  
- 🧠 Motion detection and object tracking  
- 🗑️ Waste classification using MobileNetV3  
- ⚡ Event detection and alert generation  
- 🌐 Backend API with Flask  
- 📊 Interactive dashboard (Streamlit)  
- 🗺️ GIS-based visualisation  

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

## 📁 Project Structure

```plaintext
environment-monitoring-system/
│
├── trash_v_0/                # Initial prototype version
├── trash_v_1/                # Improved version (testing + tuning)
├── trash_v_2/                # Final version (optimized pipeline)
│
├── src/
│   ├── detection/            # Motion detection logic
│   ├── tracking/             # Object tracking algorithms
│   ├── classification/       # CNN model (MobileNetV3)
│   ├── pipeline/             # Full processing pipeline
│
├── backend/
│   ├── api/                  # Flask API endpoints
│   ├── streaming/            # Event streaming (SSE)
│
├── database/
│   ├── events/               # Stored event data
│   ├── logs/                 # System logs
│
├── dashboard/
│   ├── app.py                # Streamlit dashboard
│
├── models/
│   ├── mobilenetv3/          # Trained model files
│
├── data/
│   ├── raw/                  # Raw dataset
│   ├── processed/            # Processed dataset
│
├── utils/                    # Helper functions
├── tests/                    # Testing scripts
│
├── docs/
│   ├── design/               # Design phase documents
│   ├── evaluation/           # Evaluation results
│   ├── ethics/               # Ethical analysis
│   ├── project_management/   # PID, planning docs
│
├── main.py                   # Main entry point
├── requirements.txt          # Dependencies
...
└── README.md
```

---

## 🔄 Version Control

Version control was managed using **Git**, enabling iterative development and collaboration.

| Version | Description |
|--------|------------|
| v1.0 | Initial pipeline |
| v2.0 | Motion detection implemented |
| v3.0 | Classification integrated |
| v4.0 | Backend system developed |
| v5.0 | Dashboard implemented |
| v6.0 | Final optimised system |

---

## ▶️ Installation

```bash
git clone https://github.com/your-username/illegal-dumping-detection.git
cd illegal-dumping-detection
pip install -r requirements.txt
▶️ Usage

Run the main system:

python main.py

Run the dashboard:

streamlit run dashboard/app.py
📊 Performance Targets
Accuracy: ≥ 80%
Processing Speed: 15–30 FPS
Latency: < 200 ms
⚖️ Ethical Considerations
No facial recognition used
Data minimisation applied
Transparent event logging
Designed in line with GDPR principles
🔮 Future Improvements
Integration of advanced models (e.g., YOLO)
Cloud deployment for scalability
Edge AI optimisation
Improved dataset diversity
👨‍💻 Authors
Your Team Name / Members
📄 License

This project is for academic purposes.
