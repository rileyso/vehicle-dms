# Gaze tracking for driver monitoring system

A small demo: approximate **head orientation**, **blink /
eye-closure** tracking and **eye/iris landmark overlay** from a laptop webcam
(or a video file). Built with Python, OpenCV and MediaPipe Face Landmarker.

## Screenshots

![Demo output — head pose, blink and iris overlay](screenshots/demo-1.png)

![Demo output — eye/iris landmarks and EAR readout](screenshots/demo-2.png)

## Setup

```powershell
py -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```
