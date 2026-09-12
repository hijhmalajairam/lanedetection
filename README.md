# 🚘 ADAS Command Center & Lane Detection System

![Live Status](https://img.shields.io/badge/Status-Live-brightgreen)
![Python](https://img.shields.io/badge/Python-3.x-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-5.0+-red)
![Flask](https://img.shields.io/badge/Flask-Web_Framework-black)

A full-stack Advanced Driver Assistance System (ADAS) built from scratch using Python, OpenCV, and Flask. This project processes dashcam video in real-time to detect lane lines, track vehicles, identify traffic signs, and issue audio-visual collision warnings.

### 🌐 Live Demo & Links
* **Live Website (Render):** [https://lanedetection-h4wf.onrender.com/](https://lanedetection-h4wf.onrender.com/)
* **GitHub Repository:** [hijhmalajairam/lanedetection](https://github.com/hijhmalajairam/lanedetection)

*(**Note on Live Demo:** This project is hosted on a free Render cloud server. If the site hasn't been visited in 15 minutes, the server goes to "sleep". It may take up to 50 seconds to wake up on your first visit!)*

---

## ✨ Key Features

### 1. Computer Vision Pipeline (Lane Tracking)
* **Dynamic ROI & Masking:** Slices the video to ignore the sky and the car's physical dashboard.
* **Color & Gradient Thresholding:** Uses HLS color space (to isolate white and yellow lines regardless of shadows) combined with Sobel X-gradient edge detection.
* **Bird's-Eye View (Perspective Transform):** Warps the camera angle to a top-down view to calculate road curvature.
* **Sliding Window Search:** Uses histograms and sliding boxes to trace the exact path of the left and right lane lines.
* **Polynomial Fitting:** Draws a smooth mathematical curve over the detected lane lines and calculates the vehicle's offset from the center of the road.

### 2. Deep Learning & Object Detection
* **Vehicle Tracking:** Uses a pre-trained **MobileNet-SSD** (Single Shot Detector) Caffe model to identify vehicles on the road in real-time.
* **Traffic Sign Detection:** Uses Haar Cascades to detect Stop Signs.

### 3. Safety Warning Systems
* **Lane Departure Warning System (LDWS):** Tracks the vehicle's position relative to the center of the lane. If the car drifts too far left or right, the system flashes a red warning and triggers a 750Hz audio alarm.
* **Forward Collision Warning (FCW):** Measures the bounding box size of detected vehicles ahead. If a vehicle grows too large (indicating a rapid approach), it flashes a warning and triggers a 1000Hz audio alarm.

### 4. Interactive Web UI
* **Real-time Calibration Sliders:** Adjust the horizon line, dashboard crop, and perspective width dynamically without restarting the server.
* **9 Diagnostic View Modes:** Watch the algorithm work step-by-step by switching between Edge Mask, Hough Lines, Bird's-Eye, Sliding Polynomials, Radar Map, and the Final HUD.
* **Upload Video (Beta):** Drag and drop your own `.mp4` dashcam footage into the browser to process custom videos on the fly.

---

## 🛠️ Technology Stack

* **Backend:** Python, Flask, Gunicorn
* **Computer Vision:** OpenCV (`opencv-python-headless`), Numpy
* **Machine Learning:** MobileNet-SSD (Caffe), Haar Cascades
* **Frontend:** HTML5, CSS3, JavaScript (Fetch API, Web Audio API)
* **Deployment:** Render (Cloud PaaS), GitHub

---

## 💻 How to Run Locally

If you want to run this project on your own computer (which is recommended for processing large 4K videos without cloud memory limits):

1. **Clone the repository:**
   ```bash
   git clone https://github.com/hijhmalajairam/lanedetection.git
   cd lanedetection
   ```

2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask server:**
   ```bash
   python app.py
   ```

4. **Open in Browser:**
   Go to `http://localhost:5000` in Google Chrome or Edge.

---

## 📂 Project Structure
* `app.py`: The main Flask web server, HTTP routing, and front-end HTML/JS payload.
* `src/pipeline.py`: The master Computer Vision pipeline that calls all processing steps frame-by-frame.
* `src/vision_utils.py`: Contains math functions for perspective warping and HLS/Sobel color thresholding.
* `src/lane_tracker.py`: Contains the `LaneLine` class, polynomial fitting, and sliding window logic.
* `requirements.txt`: Python dependencies for cloud deployment.
* `test_videos/`: Default testing footage.
