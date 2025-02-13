import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import torch
from ultralytics import YOLO
import roboflow
import cv2
import numpy as np
from torchvision import transforms as T
from PIL import Image

# Load MiDaS model
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)

# Define MiDaS image transformation
midas_transform = T.Compose(
    [
        T.Resize((384, 384)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),  # MiDaS expects normalized inputs
    ]
)

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
                track_id = (
                    int(det.id[0].item()) if det.id is not None else -1
                )  # Tracking ID
                x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())  # Bounding box
                bounding_boxes.append([x1, y1, x2, y2,track_id])  # Append coordinates to the list

        overlapped = find_overlapping_boxes(bounding_boxes)

        # Extract detections (boxes, class IDs, confidences, and tracking IDs)

        # Convert frame to RGB and preprocess for MiDaS
        img_resized = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_resized)  # Convert NumPy array to PIL Image
        img_tensor = midas_transform(img_pil).unsqueeze(0).to(device)

        # Estimate depth
        with torch.no_grad():
            depth_map = midas(img_tensor).squeeze().cpu().numpy()

        # Normalize depth map
        depth_map = (depth_map - depth_map.min()) / (
            depth_map.max() - depth_map.min() + 1e-6
        )

        # Resize depth map to match frame size
        depth_resized = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))

        # Apply heatmap
        depth_colored = apply_colormap(depth_resized)

        # Overlay heatmap on the original frame
        frame = cv2.addWeighted(frame, 0.6, depth_colored, 0.4, 0)
        print(depth_resized)

        if results and results[0].boxes:
            for det in results[0].boxes:
                x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())  # Bounding box
                conf = det.conf[0].item()  # Confidence score
                lowest_point = lowest_distance(depth_resized,(x1,y1,x2,y2))
                cls = int(det.cls[0].item())  # Class ID
                track_id = (
                    int(det.id[0].item()) if det.id is not None else -1
                )  # Tracking ID
                
                bounding_boxes.append([x1, y1, x2, y2,track_id])  # Append coordinates to the list

                # Define colors
                color = (0, 255, 0)  # Green bounding box
                text_color = (0, 0, 255)  # White text

                if track_id in overlapped:
                    color = (0, 0, 255)  # Red for overlapping boxes
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                else:
                    color = (0, 255, 0)  # Green for normal boxes
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Display class, confidence, and tracking ID
                label = f"ID: {track_id} | Conf: {conf:.2f} | Dist: {lowest_point:.3f}"
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    text_color,
                    3,
                )

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Display the frame with bounding boxes/show processes frame
        cv2.imshow("YOLOv8 + DeepSORT Tracking + Real-time Video Depth Heatmap", frame,)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# def calculate_iou(box1, box2):
#     """Calculate the Intersection over Union (IoU) of two bounding boxes."""
#     x1_1, y1_1, x2_1, y2_1 = box1[:4]
#     x1_2, y1_2, x2_2, y2_2 = box2[:4]

#     # Compute intersection coordinates
#     xi1 = max(x1_1, x1_2)
#     yi1 = max(y1_1, y1_2)
#     xi2 = min(x2_1, x2_2)
#     yi2 = min(y2_1, y2_2)

#     # Compute width and height of intersection
#     inter_width = max(0, xi2 - xi1)
#     inter_height = max(0, yi2 - yi1)

#     # Compute intersection and union areas
#     intersection = inter_width * inter_height
#     area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
#     area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
#     union = area1 + area2 - intersection

#     # Compute IoU
#     if union == 0:
#         return 0  # Avoid division by zero
#     return intersection / union

def calculate_coverage(box_a, box_b):
    """Calculate the coverage of box_a covered by box_b."""
    x1_a, y1_a, x2_a, y2_a = box_a  # Box A coordinates
    x1_b, y1_b, x2_b, y2_b = box_b  # Box B coordinates

    # Calculate intersection coordinates
    x_left = max(x1_a, x1_b)
    y_top = max(y1_a, y1_b)
    x_right = min(x2_a, x2_b)
    y_bottom = min(y2_a, y2_b)

    # Compute intersection area
    if x_right <= x_left or y_bottom <= y_top:
        return 0  # No overlap

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Compute area of Box A
    area_a = (x2_a - x1_a) * (y2_a - y1_a)

    return intersection_area / area_a  # Coverage percentage

def find_overlapping_boxes(bounding_boxes, coverage_threshold=0.5):
    """Find and store overlapping bounding boxes based on IoU threshold, ensuring each combination is captured in both directions."""
    num_boxes = len(bounding_boxes)
    overlapped_boxes = []

    for i in range(num_boxes):  # Loop through each box
        for j in range(num_boxes):  # Compare with all boxes including earlier ones
            if i != j:  # Avoid self-comparison
                one_id = bounding_boxes[i][4]
                two_id = bounding_boxes[j][4]
                coverage = calculate_coverage(bounding_boxes[i][:4], bounding_boxes[j][:4])
                if coverage > coverage_threshold:  # If IoU is above threshold, they overlap
                    # Add both (i, j) and (j, i) to the list
                    overlapped_boxes.append(one_id)  # Store the pair (i, j)
                    overlapped_boxes.append(two_id)  # Store the reverse pair (j, i)
                    
                    print(f"Box {one_id} overlaps with Box {two_id} (coverage: {coverage:.2f})")

    return list(set(overlapped_boxes))
            

def apply_colormap(depth_map):
    """Apply a heatmap color mapping to the depth map."""
    depth_colored = cv2.applyColorMap(
        (depth_map * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    return depth_colored

def lowest_distance(depth,bounding_box):
    x1,y1,x2,y2 = bounding_box
    depth = depth[y1:y2]
    cropped = []

    for frame in depth:
        crop = frame[x1:x2]
        cropped.extend(crop)

    lowest_point = max(cropped)
    return lowest_point

track_realtime_with_opencv("./best.pt", "20250207_151803.mp4")

# process_video_realtime("20250207_151803.mp4")
# Example: Track objects in a video
# track_with_deepsort("./best.pt", "20250207_151803.mp4")
