# Webcam-Livestream Real-Time Object Detection

# 📷 Real-Time Object Detection and Live Video Streaming Web App

## Project Description

This project is a lightweight web application for **real-time object detection and video streaming** using a webcam. Built with `aiohttp` for asynchronous web serving and powered by **YOLOv8 (You Only Look Once)** for object detection, the app processes webcam footage, detects objects in each frame, annotates them, and streams the result to a web browser in real time.

---

## 🚀 Features

- 🧠 **YOLOv8-based Object Detection**  
  Utilizes the `ultralytics` YOLOv8n model for fast and efficient object recognition.

- 🔄 **Asynchronous Streaming**  
  Leverages `asyncio` and a `ThreadPoolExecutor` to handle frame capture, processing, and encoding without blocking the event loop.

- 🌐 **Live Video Feed in Browser**  
  Streams video using `multipart/x-mixed-replace`, viewable in any modern web browser.

- 📊 **FPS Monitoring**  
  Real-time FPS (frames per second) is displayed directly on the video feed for performance monitoring.

- 🔧 **Multithreaded Execution**  
  Handles CPU-bound operations in background threads for smoother performance.

---

## 🛠️ Technologies Used

- Python 3
- [aiohttp](https://docs.aiohttp.org/) – asynchronous HTTP server
- [OpenCV](https://opencv.org/) – image and video processing
- [Ultralytics YOLOv8](https://docs.ultralytics.com/) – object detection model
- `asyncio`, `concurrent.futures` – async programming and thread pooling
- HTML + Bootstrap – responsive frontend UI

---

## 🖥️ Usage

1. **Install the required packages:**
   ```bash
   pip install aiohttp opencv-python ultralytics

2. **Run the application:**
   ```bash
   python app.py

4. **Open your browser and go to:**
  [http://localhost:8080](http://localhost:8080)

5. **View the live, annotated webcam feed in real time!**
