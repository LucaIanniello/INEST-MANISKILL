import os
import subprocess
from pathlib import Path
import cv2

# === CONFIGURATION ===
SRC_DIR = Path("/home/liannello/.maniskill/demos/StackPyramid-v1/motionplanning/video_dataset/video/valid")   # Directory containing .mp4 files
DEST_DIR = Path("/home/liannello/.maniskill/demos/StackPyramid-v1/motionplanning/video_dataset/frames/valid")  # Directory to store extracted frames
FPS = 30                                   # Base extraction fps
FRAME_INTERVAL = 3                          # Take 1 frame every 3 (i.e., 10 fps)


def extract_frames(video_path: Path, output_dir: Path):
    """Extract frames using OpenCV instead of ffmpeg."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Could not open {video_path}")
        return

    frame_count = 0
    saved_count = 0

    print(f"Extracting frames from {video_path.name} ...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save every 10th frame
        if frame_count % FRAME_INTERVAL == 0:
            saved_count += 1
            frame_path = output_dir / f"{saved_count}.png"
            cv2.imwrite(str(frame_path), frame)

        frame_count += 1

    cap.release()
    print(f"✅ Saved {saved_count} frames to {output_dir}\n")

def main():
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    videos = sorted(SRC_DIR.glob("*.mp4"))
    if not videos:
        print(f"No .mp4 files found in {SRC_DIR}")
        return

    for video_path in videos:
        video_name = video_path.stem
        output_dir = DEST_DIR / video_name
        extract_frames(video_path, output_dir)

    print("🎉 All videos processed successfully!")

if __name__ == "__main__":
    main()