import cv2
import os
from pathlib import Path

# --- Configuration ---
VIDEO_PATH = "test_imgs/video/DJI_29_D.MP4"
OUTPUT_DIR = "test_imgs/sequences/01"
FRAME_INTERVAL_SECONDS = 1

def extract_frames():
    # Setup paths
    video_path = Path(VIDEO_PATH)
    output_dir = Path(OUTPUT_DIR)
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # Initialize video capture
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {VIDEO_PATH}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    
    print(f"Video FPS: {fps}")
    print(f"Total Duration: {duration:.2f} seconds")
    
    # Calculate frame skip
    frame_step = int(fps * FRAME_INTERVAL_SECONDS)
    frame_step = 1 # Force every frame, remove to use the calculated frame step
    if frame_step < 1:
        frame_step = 1

    count = 0
    extracted_count = 0

    print(f"Starting extraction to {OUTPUT_DIR}...")

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Only save frames at the specified interval
        if count % frame_step == 0:
            # Formatting filename for temporal order (00001.jpg, 00002.jpg...)
            filename = f"{extracted_count:06d}.jpg"
            save_path = output_dir / filename
            
            # Save frame (using high quality for better feature matching)
            cv2.imwrite(str(save_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            extracted_count += 1

        count += 1

    cap.release()
    print(f"Done! Extracted {extracted_count} frames.")

if __name__ == "__main__":
    extract_frames()