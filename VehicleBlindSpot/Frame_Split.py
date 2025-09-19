import cv2
import os
from tqdm import tqdm


def extract_frames_from_folder(data_folder, output_root, frame_sampling=30):
    """
    Extracts frames from all video files in a folder and mirrors the folder structure
    in the output directory, saving extracted frames inside respective subfolders.

    Parameters:
        data_folder (str): Path to the input folder containing video files.
        output_root (str): Path to the root output folder where extracted frames will be saved.
        frame_sampling (int): Save every 'frame_sampling' frame.
    """

    # Ensure output root exists
    os.makedirs(output_root, exist_ok=True)

    video_files = []
    for root, _, files in os.walk(data_folder):
        for file in files:
            if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                video_files.append(os.path.join(root, file))

    for video_path in tqdm(video_files, desc="Processing videos"):
        relative_path = os.path.relpath(os.path.dirname(video_path), data_folder)
        output_folder = os.path.join(
            output_root,
            relative_path,
            os.path.splitext(os.path.basename(video_path))[0],
        )
        os.makedirs(output_folder, exist_ok=True)
        extract_frames(video_path, output_folder, frame_sampling)


def extract_frames(video_path, output_folder, frame_sampling=5):
    """
    Extracts frames from a video file at a given sampling rate and saves them as images.

    Parameters:
        video_path (str): Path to the input video file.
        output_folder (str): Path to save the extracted frames.
        frame_sampling (int): Save every 'frame_sampling' frame.
    """

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0  # Total frames processed
    saved_count = 0  # Frames actually saved

    with tqdm(
        total=total_frames,
        desc=f"Extracting {os.path.basename(video_path)}",
        unit="frame",
    ) as pbar:
        while True:
            ret, frame = cap.read()

            if not ret:
                break  # Break when video ends

            if frame_count % frame_sampling == 0:
                frame_filename = os.path.join(
                    output_folder, f"frame_{saved_count:06d}.jpg"
                )
                
                frame = cv2.flip(
                    frame, 1
                )  # Flip horizontally (mirror), to correct orientation
                cv2.imwrite(frame_filename, frame)
                saved_count += 1

            frame_count += 1
            pbar.update(1)

    cap.release()
    print(f"Extraction completed for {video_path}. {saved_count} frames saved.")


# Example usage
data_folder = "./data"
output_root = "extracted_frames"
frame_sampling = 30  # Save every 5th frame
extract_frames_from_folder(data_folder, output_root, frame_sampling)
