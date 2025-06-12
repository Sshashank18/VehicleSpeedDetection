# 🚗 Vehicle Speed Detection using OpenCV & dlib

This project detects moving vehicles in a video feed and estimates their speed using image processing and object tracking techniques. It utilizes OpenCV for detection and dlib for tracking, and displays the speed of each car in the output video.

---

## 📸 Features

- Detects vehicles using Haar cascade classifier.
- Tracks vehicles across frames using `dlib.correlation_tracker`.
- Estimates speed based on object displacement across frames.
- Draws bounding boxes and speed labels on moving vehicles.
- Saves the processed video with real-time detection and tracking overlays.

---

## 🧰 Requirements

Make sure you have the following installed:

```bash
pip install opencv-python
pip install cmake            # Required for dlib
pip install dlib             # You must have CMake and a C++ compiler installed
```

## Screenshots

<img src="https://github.com/Sshashank18/VehicleSpeedDetection/blob/master/Screenshots/Screenshot%20(1).png"
     style="float: left; margin-right: 10px;"/>
