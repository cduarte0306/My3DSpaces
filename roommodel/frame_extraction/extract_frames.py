# Step 1: ffmpeg frame extraction (~2-3 fps) + blur filtering (Laplacian variance).
import cv2 as cv
import numpy as np

def extract_frames(video_path: str, output_dir: str, fps: float = 2.5) -> list[str]:
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    # First, clear the output directory
    import os
    for f in os.listdir(output_dir):
        if f.endswith(".png"):
            os.remove(os.path.join(output_dir, f))

    # Set the video to 3 frames per second
    video_fps = cap.get(cv.CAP_PROP_FPS)
    frame_paths = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % int(video_fps / fps) == 0:
            frame_path = f"{output_dir}/frame_{frame_idx:06d}.png"
            cv.imwrite(frame_path, frame)
            if is_blurry(frame_path):
                os.remove(frame_path)
            else:
                frame_paths.append(frame_path)

        frame_idx += 1

    cap.release()
    return frame_paths

def is_blurry(frame_path: str, threshold: float = 20.0) -> bool:
    frame = cv.imread(frame_path, cv.IMREAD_GRAYSCALE)
    if frame is None:
        raise ValueError(f"Failed to read frame: {frame_path}")
    laplacian_var = cv.Laplacian(frame, cv.CV_64F).var()
    return laplacian_var < threshold