import logging
from dataclasses import dataclass

import cv2
from cv2.typing import MatLike, Rect

from constants import *
from frame import Frame

logger = logging.getLogger(f"{BATOMETER}.ObjectFinder")


@dataclass(frozen=True)
class Object:
    x: int
    y: int
    w: int
    h: int


class ObjectFinder:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def __init__(self) -> None:
        self.backgroundSub = cv2.createBackgroundSubtractorMOG2(
            history=500,  # no. frames to keep
            varThreshold=100,  # sensitivity of
            detectShadows=False,
        )

    def initialise(self, video: cv2.VideoCapture) -> None:
        # Priming with initial frames (background should be stable)
        logger.info("Priming background subtractor...")
        initial_frame_count = 500
        for _ in range(initial_frame_count):
            ret, frame = video.read()
            if not ret:
                break
            # Update the background model with initial frames
            self.backgroundSub.apply(frame)
        logger.info("Successfully primed background subtractor")

    def update(self, frame: Frame) -> list:
        # Create the foreground mask
        fgmask = self.backgroundSub.apply(frame.frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel)
        # Find contours on the foreground
        detections = self._get_contours(fgmask)
        return detections

    def _get_contours(self, frame: MatLike) -> list:
        contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_poly: list[None | MatLike] = [None] * len(contours)
        boundRect: list[None | Rect] = [None] * len(contours)
        detections = []
        for i, contour in enumerate(contours):
            # Get the centroid of the object
            M = cv2.moments(contour)
            if M["m00"] != 0:  # Area is non-zero
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                x, y, w, h = cv2.boundingRect(contour)
                detections.append([x, y, w, h])
                logger.debug(f"Found object x: {cx} y: {cy}")
                # cv2.drawContours(contour_frame, contour, -1, (0, 255, 0), 3)
                approx = cv2.approxPolyDP(contour, 3, True)
                contours_poly[i] = approx
                if approx is None or len(approx) < 0:
                    logger.warning("Could not approximate contour")
                    continue
                boundRect[i] = cv2.boundingRect(approx)
        return detections
