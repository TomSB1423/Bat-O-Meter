import os
import shutil
from typing import List

import cv2
import numpy as np


def visualize_tracking(detections_per_frame: List, tracked_per_frame: List, window_name: str):
    """
    Save detections and tracked IDs for each frame as images in a directory.
    Images for each test are saved in artifacts/visualize_tracking/{window_name}/ by default.
    The subdirectory is cleared at the start of each test run.
    Returns the directory where images are saved.
    """
    height, width = 400, 400
    save_dir = os.path.join(".artifacts", "visualize_tracking", window_name)
    # Clear the subdirectory if it exists
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    for frame_idx, (detections, tracked) in enumerate(zip(detections_per_frame, tracked_per_frame)):
        img = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background
        cv2.putText(
            img, str(frame_idx + 1), (width - 25, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
        )
        # Draw detections (red)
        for det in detections:
            cv2.rectangle(img, (det.x, det.y), (det.x + det.width, det.y + det.height), (0, 0, 255), 1)
        # Draw tracked objects (green) and IDs
        for obj in tracked:
            cv2.rectangle(img, (obj.x, obj.y), (obj.x + obj.width, obj.y + obj.height), (0, 255, 0), 2)
            cv2.putText(img, str(obj.id), (obj.x, obj.y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        img_path = os.path.join(save_dir, f"frame_{frame_idx+1}.png")
        cv2.imwrite(img_path, img)
