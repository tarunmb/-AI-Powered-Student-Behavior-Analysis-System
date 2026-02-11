# 🎓 AI-Powered Student Behavior Analysis System

## 🚀 Problem Statement

Educators often struggle to monitor classroom engagement and behavioral patterns at scale. Manual observation is subjective, inconsistent, and difficult to analyze over time.

This project builds a real-time AI-powered behavior analysis system that uses computer vision and deep learning to detect student activities, enabling automated engagement monitoring and data-driven insights.

---

## 💡 Solution Overview

The system processes live webcam feeds or uploaded media, detects student behaviors using a YOLOv8 deep learning model, and presents results through a web-based interface.

Pipeline flow:

Video/Image Input → YOLO Behavior Detection → Frame Processing → Insight Visualization

This architecture demonstrates how AI-driven computer vision can support scalable behavioral analytics.

---

## 🔍 Key Features

* 🎥 Real-time behavior detection using YOLOv8
* 📸 Image and video analysis pipeline
* 🔄 Multi-threaded processing for performance optimization
* 📊 Automated behavior tracking and analysis
* 🌐 Web interface for easy interaction
* 📱 Responsive UI for cross-device access

---

## 🧠 Detected Behaviors

The system identifies classroom engagement patterns such as:

* Using laptop
* Using mobile phone
* Reading
* Writing
* Looking away
* Sleeping
* Laughing

These classifications provide actionable insights into student activity trends.

---

## 🏗 System Architecture

1. Media input captured via webcam or upload
2. YOLOv8 model performs behavior detection
3. Frame-level processing and optimization
4. Results rendered through Flask web interface
5. Outputs stored for review and analysis

Data Flow:

Media Input → Detection Model → Processing Layer → Web Visualization

---

## ⚙ Tech Stack

* **Language:** Python
* **Framework:** Flask
* **Computer Vision:** OpenCV
* **Deep Learning:** YOLOv8 (Ultralytics)
* **Frontend:** HTML/CSS/JS

---

## 📊 System Capabilities

* Real-time inference pipeline
* Performance optimization via multi-threading
* Efficient frame handling for smoother processing
* Structured output generation

---

## ▶ How to Run

1. Clone the repository
   git clone <repo-url>

2. Create virtual environment
   python -m venv venv
   activate environment

3. Install dependencies
   pip install -r requirements.txt

4. Start the application
   python app.py

5. Open browser → http://localhost:5001

Upload media or enable real-time analysis to begin detection.

---

## 📁 Project Structure

app.py → Core application logic
models/ → YOLO model files
templates/ → Web interface
static/ → Frontend assets
data/ → Uploaded media
output/ → Processed results

---

## 🔮 Future Improvements

* Advanced engagement analytics dashboard
* Database-backed behavioral tracking
* Model accuracy enhancements
* Cloud deployment pipeline
* Real-time alert system

---

## 📌 Key Learnings

* Real-time computer vision pipeline design
* Deep learning inference integration
* Performance optimization strategies
* Web-based AI system deployment

---


## 🤝 Acknowledgments

YOLOv8 — Ultralytics
Flask — Web framework
OpenCV — Image processing
