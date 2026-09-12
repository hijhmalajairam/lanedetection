A
Computer Vision Course Assignment Project Report
On

**FittingLines: Development of an Automated Lane Departure Warning System using Line Fitting and Object Detection**

Submitted by:
**HIJHMALA JAIRAM**
**24STUCHH010669**
**BTECH-AI&DS**

Faculty Name:
**Dr. P. Chakradhar**
Sr. Asst. Prof., Dept. of CSE

Faculty of Science & Technology
IFHE University, Hyderabad
Sept 2026

---

## ABSTRACT

The rapid advancement of intelligent transportation systems has made Advanced Driver Assistance Systems (ADAS) a critical component in preventing traffic accidents. This project presents the development of a real-time, web-based ADAS Command Center that performs automated lane detection, vehicle tracking, and collision warning. The core methodology relies on classical computer vision techniques, specifically advanced line fitting via sliding window algorithms on a perspective-warped "bird's-eye" view of the road. To ensure robustness against varying lighting conditions, the image preprocessing pipeline utilizes a combination of HLS color space thresholding and Sobel X-gradient edge detection. 

In addition to lane departure warning systems (LDWS), the architecture integrates a MobileNet-SSD deep learning model for real-time forward collision warning (FCW) and Haar Cascades for traffic sign detection. The entire backend is served via a Flask web application, streaming the processed video output to a custom HTML/JS frontend Heads-Up Display (HUD). The user interface features dynamic calibration sliders, allowing users to manually adjust horizon lines and perspective width in real time. The final system demonstrates high accuracy in tracking road curvature and issuing immediate audio-visual alerts upon detecting unsafe lane departures or imminent forward collisions, proving the viability of deploying lightweight ADAS algorithms on web architectures.

**Keywords:** Advanced Driver Assistance Systems (ADAS), Computer Vision, OpenCV, Lane Departure Warning System (LDWS), Sliding Window Algorithm, Polynomial Line Fitting, Object Detection.

---

## TABLE OF CONTENTS

| S. No. | Section Title | Page No. |
| :--- | :--- | :--- |
| 1 | Introduction | 3 |
| 2 | Case Design / Algorithm Implementation | 4 |
| 3 | Results and Discussion | 7 |
| 4 | Conclusion | 9 |
| 5 | References | 10 |
| 6 | Appendix | 11 |

---

## 1. Introduction

### Background of the Project
Driver error remains the leading cause of traffic accidents globally. Advanced Driver Assistance Systems (ADAS) aim to mitigate this by automating specific driving tasks and alerting drivers to potential hazards. While commercial ADAS systems rely on expensive, proprietary hardware, this project explores the feasibility of building a robust, software-based ADAS that processes standard dashcam video feeds using open-source Computer Vision libraries.

### Problem Definition
Detecting lane lines in real-time is computationally challenging due to dynamic environmental variables, including shadows, glare, faded paint, and varying road curvatures. Furthermore, classical lane detection scripts are often static and difficult to tune. The problem lies in creating a dynamic pipeline that can isolate lane lines under varying conditions while simultaneously performing object detection without exceeding the computational limits of standard hardware.

### Objectives
1. To develop a robust computer vision pipeline utilizing perspective warping and polynomial line fitting to track lane curvature.
2. To implement a Lane Departure Warning System (LDWS) and a Forward Collision Warning (FCW) mechanism.
3. To integrate deep learning models (MobileNet-SSD) for real-time vehicle identification.
4. To build an interactive web-based dashboard allowing real-time calibration of camera parameters and providing diagnostic visualizations of the algorithm.

---

## 2. Case Design / Algorithm Implementation

### Datasets, Tools, Languages, and Frameworks Used
* **Programming Language:** Python 3, JavaScript
* **Computer Vision Library:** OpenCV (`opencv-python-headless`), NumPy
* **Machine Learning:** MobileNet-SSD (Caffe framework), Haar Cascades (XML)
* **Web Framework:** Flask, Gunicorn
* **Frontend:** HTML5, CSS3, Web Audio API
* **Deployment:** Render (Cloud PaaS), GitHub

### Modules / Components Description
1. **Image Preprocessing Module:** Converts the raw BGR frame to HLS color space to isolate white and yellow lane lines. It computes the Sobel X-gradient to detect vertical edges, combining both into a single binary mask.
2. **Perspective Transformation Module:** Calculates a homography matrix dynamically based on user UI sliders. It warps the masked image into a top-down "bird's-eye" view.
3. **Lane Tracking Module:** Computes a column-wise histogram of the warped image to find the lane bases. It utilizes a sliding window algorithm to trace the non-zero pixels upwards, fitting a second-degree polynomial ($f(y) = Ay^2 + By + C$) to mathematically map the road's curve.
4. **Safety & Alert Module:** Analyzes the fitted polynomials to calculate the vehicle's drift from the lane center. It triggers the Web Audio API on the frontend if a lane departure or forward collision threshold is crossed.

### Flowcharts and Algorithms
**Algorithm: Sliding Window Lane Detection**
1. Generate histogram of the bottom half of the binary warped image.
2. Identify left and right peaks in the histogram as starting x-coordinates.
3. Divide the image into $N$ horizontal windows.
4. For each window, identify non-zero pixels within a defined margin.
5. Append pixel coordinates to active arrays and recenter the window based on the mean pixel position.
6. Extract all detected pixels and fit a 2nd-degree polynomial using `numpy.polyfit`.

### Code Snippets
*Dynamic Perspective Warping:*
```python
def get_perspective_transform_matrices(img_shape, top_y=0.63, bottom_y=0.95, top_w=0.10):
    h, w = img_shape[:2]
    src = np.float32([
        [w * (0.5 - top_w/2), h * top_y],
        [w * (0.5 + top_w/2), h * top_y],
        [w * 0.85, h * bottom_y],
        [w * 0.15, h * bottom_y]
    ])
    dst = np.float32([[w*0.25, 0], [w*0.75, 0], [w*0.75, h], [w*0.25, h]])
    return cv2.getPerspectiveTransform(src, dst)
```

### Screen-wise Explanation
The Web UI consists of three primary panels:
1. **Live ADAS Camera Feed:** Displays the processed MJPEG video stream overlaid with a HUD, radar tracking, and collision alerts.
2. **Control Panel (Upload & Reset):** Allows users to upload custom dashcam videos and automatically assigns optimized calibration presets based on the file.
3. **Camera Calibration & Display Modes:** Features HTML sliders that directly manipulate the Python backend's ROI variables. It includes 9 buttons to toggle through diagnostic view modes (e.g., Edge Mask, Hough Lines, Optical Flow).

---

## 3. Results and Discussion

### Output Screens and Results
The system successfully identifies lane curvatures and projects a green polygonal area onto the drivable road surface. When the vehicle drifts >0.4m from the center, the HUD flashes "LANE DEPARTURE WARNING" and sounds an auditory alarm. The diagnostic "Mini-Monitors" on the right side of the HUD successfully visualize the intermediate binary and sliding-window stages in real-time.

### Performance Analysis
Using standard OpenCV processing, the system maintains a frame rate of approximately 25-30 FPS locally. When deployed to a 512MB RAM cloud instance via Gunicorn (`--threads 4`), the application remains highly responsive, effectively handling concurrent HTTP requests for UI adjustments without blocking the MJPEG stream.

### Comparison with Existing Systems
Unlike standard academic scripts that output a static MP4 file, this system provides a live, interactive web dashboard. It solves the issue of hardcoded camera angles by allowing users to physically drag sliders to match the horizon and hood-crop of completely unknown camera feeds dynamically.

### Challenges Faced and How They Were Solved
1. **Cloud Server UI Blocking:** The Flask server initially froze when UI buttons were clicked because the infinite video generator held the single Gunicorn worker hostage. **Solution:** Configured Gunicorn to run with `--threads 4 --timeout 0`, allowing asynchronous request handling.
2. **Browser Audio Autoplay Policy:** Browsers blocked the ADAS warning beeps because audio requires explicit user interaction. **Solution:** Implemented a full-screen "Initialization Overlay" requiring the user to click the screen to launch the app, unlocking the `AudioContext`.
3. **Missing GUI on Cloud Servers:** `cv2.imshow` caused the server to crash because cloud instances lack physical monitors. **Solution:** Substituted standard OpenCV with `opencv-python-headless` and streamed raw byte arrays over HTTP.

---

## 4. Conclusion

### Summary of What Was Achieved
This project successfully developed a full-stack, cloud-deployable Advanced Driver Assistance System. By combining mathematical line fitting with deep learning object detection, the system accurately maps road geometries and warns the user of hazards. The creation of an interactive Web HUD elevates the project from a standard script to a usable software application.

### Limitations
1. The sliding window algorithm relies heavily on clearly visible road markings. Performance degrades significantly at night or on dirt roads.
2. The perspective matrix (horizon line) must be manually calibrated for different vehicles, as camera height and mounting angles vary.

### Possible Improvements
1. Implementing an automated horizon-detection algorithm to remove the need for manual slider calibration.
2. Upgrading the classical lane detection pipeline to an end-to-end Deep Learning semantic segmentation model (e.g., U-Net or SegNet) for increased robustness in poor weather.

---

## 5. References

[1] G. Bradski, "The OpenCV Library," *Dr. Dobb's Journal of Software Tools*, vol. 25, pp. 120-125, Nov. 2000.
[2] A. Grinberg, *Flask Web Development*, 2nd ed. Sebastopol, CA: O'Reilly Media, 2018.
[3] W. Liu, D. Anguelov, D. Erhan, et al., "SSD: Single Shot MultiBox Detector," in *Proc. European Conf. on Computer Vision (ECCV)*, Amsterdam, Netherlands, 2016, pp. 21-37.
[4] "OpenCV Documentation," OpenCV.org, 2024. [Online]. Available: https://docs.opencv.org [Accessed: Sept. 10, 2026].

---

## 6. Appendix

**GitHub Repository:**
[https://github.com/hijhmalajairam/lanedetection](https://github.com/hijhmalajairam/lanedetection)

**Live Deployment URL:**
[https://lanedetection-h4wf.onrender.com/](https://lanedetection-h4wf.onrender.com/)

**User Manual:**
1. Navigate to the Live URL.
2. Click the black overlay to initialize the system and enable audio.
3. Observe the Main Video processing on the HUD.
4. (Optional) Drag and drop a custom `.mp4` dashcam video into the "Upload Video" box.
5. Use the "Camera Calibration" sliders to align the crop lines with the physical horizon of the uploaded video.
