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

## 🏗️ System Architecture

```markdown
![System Architecture](diagrams/System%20Architecture%20v2.png)
```

---

## 🧩 Project Architecture Diagram

```markdown
![Project Architecture](diagrams/Project%20Architecture%20Diagram.png)
```

---

## 🛠️ Tech Stack

* **Programming:** Python
* **Computer Vision:** OpenCV
* **AI Framework:** TensorFlow / PyTorch
* **Backend:** Flask
* **Frontend:** Streamlit
* **Version Control:** Git & GitHub

---

## ▶️ Installation

```bash
git clone https://github.com/tsabesoo/Environment-monitoring-system.git
cd Environment-monitoring-system
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the main system:

```bash
python main.py
```

Run the dashboard:

```bash
python dashboard/app.py
```

---

## 📁 Project Structure

```bash
environment-monitoring-system/
│
├── dashboard/               # Streamlit dashboard (added in v3)
├── data/                    # Dataset (raw + processed)
├── diagrams/                # System diagrams (architecture, intent, project design)
├── docs/                    # Design, evaluation, ethics, PM documents
├── logs/                    # System logs
├── models/                  # Trained models (MobileNetV3)
├── src/                     # Core system modules
├── training/                # Model training scripts
│
├── trash_v_0/               # Initial prototype
├── trash_v_1/               # Motion detection added
├── trash_v_2/               # Object tracking introduced
├── trash_v_3/               # Dashboard integration
├── trash_v_4/               # Backend system added
├── trash_v_5/               # Event detection improvements
├── trash_v_6/               # Model optimisation
├── trash_v_7/               # Final optimised system
├── main.py                  # Main system entry point
├── README.md
```

---

## 🔄 Version Evolution

```markdown
| Version | Description |
|--------|------------|
| v0.0 | Initial concept and basic pipeline setup |
| v1.0 | Motion detection implemented |
| v2.0 | Object tracking introduced |
| v3.0 | Dashboard (Streamlit) integrated |
| v4.0 | Backend system (Flask API) developed |
| v5.0 | Event detection and alert system refined |
| v6.0 | Model optimisation and performance tuning |
| v7.0 | Final optimised real-time system |

```

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

