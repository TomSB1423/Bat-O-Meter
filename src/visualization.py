from typing import List

import cv2
import numpy as np


def visualize_tracking(detections_per_frame: List, tracked_per_frame: List, window_name: str) -> None:
    """
    Visualize detections and tracked IDs for each frame using OpenCV.
    Red rectangles: detections, Green rectangles: tracked objects, Blue text: IDs.
    """
    WINDOW_HEIGHT = 400
    WINDOW_WIDTH = 400
    for frame_idx, (detections, tracked) in enumerate(zip(detections_per_frame, tracked_per_frame)):
        img = np.ones((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8) * 255  # Larger white background
        cv2.putText(
            img,
            str(frame_idx + 1),
            (WINDOW_HEIGHT - 25, WINDOW_WIDTH - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
        )
        # Draw detections (red)
        for det in detections:
            cv2.rectangle(img, (det.x, det.y), (det.x + det.width, det.y + det.height), (0, 0, 255), 1)
        # Draw tracked objects (green) and IDs
        for obj in tracked:
            cv2.rectangle(img, (obj.x, obj.y), (obj.x + obj.width, obj.y + obj.height), (0, 255, 0), 2)
            cv2.putText(img, str(obj.id), (obj.x, obj.y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.imshow(window_name, img)
        cv2.waitKey(500)  # Show each frame for 500ms
    cv2.destroyAllWindows()
