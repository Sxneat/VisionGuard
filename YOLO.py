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

        bounding_boxes = []  # Initialize list before the loop

        # Run YOLOv8 + DeepSORT
        results = model.track(frame, persist=True, tracker="bytetrack.yaml")

        if results and results[0].boxes:
            for det in results[0].boxes:
                x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())  # Bounding box
                bounding_boxes.append([x1, y1, x2, y2])  # Append coordinates to the list

        find_overlapping_boxes(bounding_boxes)

        # Extract detections (boxes, class IDs, confidences, and tracking IDs)
        if results and results[0].boxes:
            for det in results[0].boxes:
                x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())  # Bounding box
                conf = det.conf[0].item()  # Confidence score
                cls = int(det.cls[0].item())  # Class ID
                track_id = (
                    int(det.id[0].item()) if det.id is not None else -1
                )  # Tracking ID

                # Define colors
                color = (0, 255, 0)  # Green bounding box
                text_color = (255, 255, 255)  # White text

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Display class, confidence, and tracking ID
                label = f"ID: {track_id} | Conf: {conf:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    text_color,
                    2,
                )

        # Display the frame with bounding boxes
        cv2.imshow("YOLOv8 + DeepSORT Tracking", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

def calculate_iou(box1, box2):
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Compute intersection coordinates
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    # Compute width and height of intersection
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)

    # Compute intersection and union areas
    intersection = inter_width * inter_height
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    # Compute IoU
    if union == 0:
        return 0  # Avoid division by zero
    return intersection / union

def find_overlapping_boxes(bounding_boxes, iou_threshold=0.5):
    """Find and print overlapping bounding boxes based on IoU threshold."""
    num_boxes = len(bounding_boxes)

    for i in range(num_boxes):  # Loop through each box
        for j in range(i + 1, num_boxes):  # Compare it with all later boxes
            iou = calculate_iou(bounding_boxes[i], bounding_boxes[j])

            if iou > iou_threshold:  # If IoU is above threshold, they overlap
                print(f"Box {i + 1} overlaps with Box {j + 1} (IoU: {iou:.2f})")


track_realtime_with_opencv("./best.pt", "20250207_151803.mp4")


# Example: Track objects in a video
# track_with_deepsort("./best.pt", "20250207_151803.mp4")
