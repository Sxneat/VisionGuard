import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import cv2
import torch
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


def apply_colormap(depth_map):
    """Apply a heatmap color mapping to the depth map."""
    depth_colored = cv2.applyColorMap(
        (depth_map * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    return depth_colored


def process_video_realtime(video_path):
    """Process a video file and display depth heatmap in real-time."""

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

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
        blended = cv2.addWeighted(frame, 0.6, depth_colored, 0.4, 0)

        # Show the processed frame
        cv2.imshow("Real-time Video Depth Heatmap", blended)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# Run real-time depth estimation
if __name__ == "__main__":
    process_video_realtime("20250207_151803.mp4")

