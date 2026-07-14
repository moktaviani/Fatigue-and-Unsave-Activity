# Fatigue and Unsafe Activity Detection

A real-time computer vision system that monitors a driver through a webcam, video file, or image to detect signs of **fatigue/drowsiness** (closed eyes, yawning) and **unsafe activities** (smoking, vaping, phone use) while driving. The system raises visual and audio alerts when these conditions are detected.

## Overview

This project uses two separate YOLOv11n object detection models running in parallel:

1. [**Fatigue Detection Model**](https://github.com/moktaviani/Fatigue-Detection-Model-using-YOLOv11) — detects eye and mouth states to identify drowsiness and yawning.
2. **Unsafe Activity Detection Model** — detects smoking, vaping, and phone usage while driving.

When fatigue or unsafe behavior is detected consistently over a configurable time window, the system displays an on-screen warning and plays an audio alert.

## Features

- Real-time detection from webcam, video files, or static images
- Drowsiness detection based on closed-eye duration and event frequency
- Yawning detection based on open-mouth duration and event frequency
- Unsafe activity detection for smoking, vaping, and phone use
- Configurable confidence thresholds for each model
- Audio alerts for fatigue and unsafe activity events, with cooldown to avoid spamming
- On-screen status panel showing live counters, durations, and alert messages
- FPS counter and system resource monitor (CPU/memory) overlay
- Automatic saving of annotated output video/image
- Command-line flags to disable alerts or enable debug logging

## Project Structure

```
.
├── main.py             # Entry point: CLI, detection loop, alert logic, drawing
├── config.py           # Model paths, thresholds, class names, colors, alert settings
├── event_tracker.py    # Tracks duration/frequency of fatigue events (eye/mouth)
├── utils.py            # FPS counter, resource monitor, drawing helpers
├── models/             # Trained YOLOv11n model weights (.pt files)
└── requirements.txt    # Python dependencies
```

## Requirements

- Python 3.9+
- Dependencies listed in `requirements.txt`:
  - `ultralytics>=8.3.0`
  - `opencv-python`
  - `pygame`
  - `psutil`
  - `torch`
  - `numpy`

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/moktaviani/Fatigue-and-Unsave-Activity.git
   cd Fatigue-and-Unsave-Activity
   ```

2. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Make sure the trained model weights are placed in the `models/` folder:
   - `models/fatigue_best.pt`
   - `models/unsafe_best.pt`

## Usage

Run detection from a webcam:
```
python main.py --source webcam
```

Run detection on a video file:
```
python main.py --source path/to/video.mp4
```

Run detection on an image:
```
python main.py --source path/to/image.jpg
```

### Optional Arguments

| Argument | Description |
|---|---|
| `--fatigue-model` | Path to the fatigue detection `.pt` model (default: `models/fatigue_best.pt`) |
| `--unsafe-model` | Path to the unsafe activity detection `.pt` model (default: `models/unsafe_best.pt`) |
| `--no-fatigue-alert` | Disable audio alert for fatigue/drowsiness |
| `--no-unsafe-alert` | Disable audio alert for unsafe activity |
| `--debug` | Print all unsafe activity detections for every frame |

While a webcam or video stream is running:
- Press `q` to quit
- Press `r` to reset the event counters

Output files (annotated video or image) are automatically saved to the `output/` directory.

## Configuration

Key parameters can be adjusted in `config.py`, including:

- `CONF_THRESHOLD_FATIGUE` / `CONF_THRESHOLD_UNSAFE` — detection confidence thresholds
- `FATIGUE_CLASS_NAMES` — class labels used by the fatigue model (`closed_eye`, `closed_mouth`, `open_eye`, `open_mouth`)
- `MOUTH_OPEN_DURATION`, `MOUTH_OPEN_EVENT_LIMIT` — thresholds for triggering a "fatigued" state from yawning
- `EYE_CLOSED_DURATION`, `EYE_CLOSED_EVENT_LIMIT` — thresholds for triggering a "drowsy" state from closed eyes
- `UNSAFE_ALERT_CLASSES` — classes that trigger an unsafe activity alert (`smoking`, `vaping`, `Phone`)
- `UNSAFE_ALERT_COOLDOWN` — minimum time between repeated unsafe activity alerts
- `OUTPUT_DIR` — directory where annotated results are saved

## Model

Both the fatigue detection model and the unsafe activity detection model were trained using **YOLOv11n** (Ultralytics).

## Datasets

The models were trained using the following datasets from Roboflow Universe:

**Fatigue Detection**
- [Fatigue Detection Dataset](https://universe.roboflow.com/athena-ba1wv/fatigue-detection-loo95/dataset/1)
- [Tired Detect Dataset](https://universe.roboflow.com/smoke-tcfs5/tired-detect/dataset/3)

**Unsafe Activity Detection**
- [Phone Use Dataset](https://universe.roboflow.com/nakarin-samon/phone-use/dataset/7)
- [Smoking Vape Detection Dataset](https://universe.roboflow.com/livestreamdetection/smoking-vape-detection-7nvky/dataset/1)

## Authors

- Maria Oktaviani
- Kalya Andriana
