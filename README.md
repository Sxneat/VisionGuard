# YoloBlindspot
Motorcycle Blind-Spot Vehicle Detection

A real-time motorcycle blind-spot detection program using YOLOv8 for object detection, ByteTrack for tracking, and MiDaS for monocular depth estimation. The system highlights nearby vehicles, estimates their relative distance, and flags overlapping detections in blind-spot zones.

Overview

This project combines computer vision models into a practical safety system for motorcycles:

YOLOv8: real-time vehicle detection with bounding boxes and confidence scores.

ByteTrack / DeepSORT: consistent multi-object tracking with unique IDs.

MiDaS: monocular depth estimation to approximate distances.

Custom overlap detection: coverage-based overlap checks to identify vehicles in critical blind-spot zones.

OpenCV visualization: bounding boxes, depth heatmap overlays, and annotated labels (ID, confidence, distance).

Features

Vehicle detection using pretrained or custom-trained YOLOv8 models.

Dataset integration with Roboflow for easy downloads and training.

Depth estimation heatmap overlay for distance awareness.

Blind-spot overlap detection with customizable thresholds.

Real-time inference from video streams or recorded files.

Requirements

Python 3.8+

PyTorch

Ultralytics YOLO (pip install ultralytics)

OpenCV (pip install opencv-python)

TorchVision

NumPy

Roboflow (pip install roboflow)

4. Example Output

Each bounding box displays:

ID: unique tracking ID

Conf: YOLO detection confidence

Dist: normalized distance from MiDaS

Boxes are green by default and turn red if overlap is detected.

How It Works

Detection: YOLOv8 predicts bounding boxes and class scores.

Tracking: ByteTrack/DeepSORT assigns unique IDs for consistent tracking.

Depth Estimation: MiDaS generates a depth map, normalized and resized to match the video frame.

Blind-spot Logic: Coverage-based overlap checks (find_overlapping_boxes) flag vehicles in blind-spot areas.

Visualization: Bounding boxes, heatmaps, and text overlays are displayed in real time using OpenCV.

Functions Overview

download_dataset(): Fetch dataset from Roboflow.

train_yolov8(): Train YOLOv8 model with dataset YAML.

track_with_deepsort(): Run YOLOv8 + DeepSORT on video.

track_realtime_with_opencv(): Run YOLOv8 + ByteTrack + MiDaS depth estimation in real time.

find_overlapping_boxes(): Detect overlapping tracked vehicles.

lowest_distance(): Compute closest depth inside a bounding box.

apply_colormap(): Visualize depth as a heatmap.

Safety Notes

This program is for research and prototyping only.

Do not rely solely on it while riding.

Always comply with local laws and test in controlled environments first.