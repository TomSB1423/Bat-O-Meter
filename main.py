import cv2
import logging
from window import *
from utils import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Bat-O-Meter")


if __name__ == "__main__":
    VIDEO_PATH = r"C:\Users\TSBus\OneDrive\Bat-O-Meter\videos\batInThermal.mp4"
    video, width, height, fps = load_video(VIDEO_PATH)
    window = Window()

    backgroundSub = cv2.createBackgroundSubtractorMOG2(
        history=1000, varThreshold=50, detectShadows=False
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Initialize a list to store bird centroids across frames
    centroids = []
    while video.isOpened():
        frame_no = int(video.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = video.read()
        og_frame = frame.copy()
        frame_time = caluclate_time(frame_no, fps)
        if not ret:
            break

        # Apply background subtractor to get the foreground maskv
        fgmask = backgroundSub.apply(frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        # Find contours in the mask
        contours, _ = cv2.findContours(
            fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            # Get the centroid of the bird
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))
                logger.info(
                    f"Found bird x: {cx} y: {cy} frame: {frame_no} time: {frame_time}"
                )
                cv2.drawContours(frame, contour, -1, (0, 255, 0), 3)
                window.display_images_side_by_side_with_overlay(
                    og_frame, fgmask, frame, f"f: {frame_no} t: {frame_time}"
                )
