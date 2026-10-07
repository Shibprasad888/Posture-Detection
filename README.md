# Posture Detection

A real-time posture monitoring application built with Python, OpenCV, MediaPipe Pose, and NumPy. It tracks posture in a webcam feed, provides live posture scoring, alerts, calibration, session analytics, and report export.

## Features
- Live pose detection and skeleton overlay
- Posture status classification with highlighted regions
- Adaptive scoring and smart posture recommendations
- Calibration mode for personalized thresholds
- Break reminders and session alerts
- Session analytics and CSV export
- Screenshot capture and report generation

## Requirements
- Python 3.11
- OpenCV
- MediaPipe
- NumPy
- psutil

## Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python main.py
```

## Controls
- `q` - quit the application
- `s` - capture a screenshot
- `c` - start or stop calibration mode

## Output Files
- Screenshots are saved to the screenshots directory
- Session reports and CSV exports are saved to the reports directory
