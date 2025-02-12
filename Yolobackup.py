import os
import torch
from ultralytics import YOLO
import roboflow
import cv2

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 1: Download Dataset from Roboflow

def download_dataset(api_key: str, workspace: str, project: str, version: int, location: str = "./data"):
    """Download dataset from Roboflow and unzip it."""
    rf = roboflow.Roboflow(api_key)
    dataset = rf.workspace(workspace).project(project).version(version).download("yolov8", location)
    return dataset.location

# Step 2: Train YOLOv8 Model

def train_yolov8(data_yaml: str, model_type: str = "yolov8s.pt", epochs: int = 50, imgsz: int = 640):
    """Train YOLOv8 model."""
    model = YOLO(model_type)
    model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, device=device)
    return model

# Step 3: Track Objects with DeepSORT

def track_with_deepsort(model_path: str, video_path: str, output_path: str = "./output.mp4"):
    """Run YOLOv8 + DeepSORT for object tracking."""
    model = YOLO(model_path)
    results = model.track(source=video_path, tracker="bytetrack.yaml", save=True)
    print(f"Tracking complete. Output saved at {output_path}")
    return results

def track_realtime_with_opencv(model_path: str, video_path: str):
    """Run YOLOv8 + DeepSORT for object tracking with real-time display using OpenCV."""
    
    model = YOLO(model_path)  # Load trained YOLOv8 model
    cap = cv2.VideoCapture(video_path)  # Open video file
    
    if not cap.isOpened():
        print("Error: Cannot open video file!")
        return

    while cap.isOpened():
        ret, frame = cap.read()  # Read a frame
        if not ret:
            break  # Stop if end of video

        results = model.track(frame, persist=True, tracker="bytetrack.yaml")  # Run YOLOv8 + DeepSORT
        annotated_frame = results[0].plot()  # Draw bounding boxes & tracking IDs

        cv2.imshow("YOLOv8 + DeepSORT Tracking", annotated_frame)  # Display frame

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example Usage:
track_realtime_with_opencv("./best.pt", "20250207_151803.mp4")

# Example: Track objects in a video
# track_with_deepsort("./best.pt", "20250207_151803.mp4")