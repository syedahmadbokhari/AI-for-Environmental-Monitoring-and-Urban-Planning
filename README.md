# 🧠 AI-Powered Environment-Monitoring-System

An AI-based real-time monitoring system that detects illegal dumping using CCTV footage and computer vision techniques. The system transforms traditional surveillance into an automated, intelligent solution for smart city environments.

---

## 🚀 Overview

This project implements a **multi-stage computer vision pipeline** to detect illegal dumping events in real time. It combines motion detection, object tracking, and deep learning classification to improve accuracy and reduce false positives.

The system provides:
- 📡 Real-time monitoring  
- 🚨 Automated alerts  
- 📊 Data visualisation and analytics  

---

## 🏗️ System Architecture

The system consists of the following components:

- 📷 **CCTV Input** – Captures live video streams  
- 🧠 **AI Processing Pipeline** – Motion detection, object tracking, classification (MobileNetV3)  
- ⚠️ **Event Generation** – Creates structured event data (timestamp, confidence)  
- 🖥 **Backend System (Flask)** – API handling and data processing  
- 💾 **Database** – Stores events and logs  
- 📊 **Dashboard (Streamlit)** – Displays alerts, trends, and insights  

---

## ⚙️ Features

- Real-time video processing (15–30 FPS)  
- Motion detection and object tracking  
- Waste classification using MobileNetV3  
- Automated event generation  
- Backend API integration  
- Interactive dashboard with analytics  
- GIS-based visualisation (optional)  

---

## 🧠 Technology Stack

- **Python**  
- **OpenCV** – video processing  
- **TensorFlow / PyTorch** – deep learning  
- **Flask** – backend API  
- **Streamlit** – dashboard UI  

---

## 🔄 How It Works

1. Capture video frames from CCTV  
2. Detect motion in the scene  
3. Track moving objects  
4. Identify stationary objects  
5. Classify objects as waste/non-waste  
6. Generate event with metadata  
7. Send data to backend and store  
8. Display alerts and analytics on dashboard  

---

## 📊 Performance

- Accuracy: ~80%  
- Processing Speed: 15–30 FPS  
- Latency: <200 ms  
illegal-dumping-detection/
│
├── src/
│   ├── detection/
│   ├── classification/
│   ├── pipeline/
│   ├── backend/
│   ├── database/
│   └── utils/
│
├── dashboard/
├── models/
├── data/
├── tests/
│
├── docs/
│   ├── project_management/
│   ├── design/
│   ├── evaluation/
│   ├── ethics/
│   └── report/
│
├── main.py
├── requirements.txt
└── README.md
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
python main.py

Run dashboard:

streamlit run dashboard/app.py
