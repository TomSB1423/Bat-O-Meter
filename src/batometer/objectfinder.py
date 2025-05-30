import logging
from typing import List

import cv2
from cv2.typing import MatLike

from .constants import BATOMETER
from .detectionObject import Detection, Point
from .frame import Frame

logger = logging.getLogger(f"{BATOMETER}.ObjectFinder")


class ObjectFinder:
    """
    Detects moving objects in video frames using background subtraction and contour detection.
    """

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def __init__(self) -> None:
        """
        Initializes the background subtractor for object detection.
        """
        self.backgroundSub = cv2.createBackgroundSubtractorMOG2(
            history=500,  # no. frames to keep
            varThreshold=100,  # sensitivity of
            detectShadows=False,
        )

    def initialise(self, video: cv2.VideoCapture) -> None:
        """
        Primes the background subtractor with initial frames to stabilize the background model.

        Args:
            video (cv2.VideoCapture): The video capture object to read frames from.
        """
        logger.info("Priming background subtractor...")
        initial_frame_count = 500
        for _ in range(initial_frame_count):
            ret, frame = video.read()
            if not ret:
                break
            # Update the background model with initial frames
            self.backgroundSub.apply(frame)
        logger.info("Successfully primed background subtractor")

    def update(self, frame: Frame) -> set[Detection]:
        """
        Updates the object finder with a new frame and returns detected objects.

        Args:
            frame (Frame): The current video frame.

        Returns:
            List[DetectionObject]: List of detected objects in the frame.
        """
        # Create the foreground mask
        fgmask = self.backgroundSub.apply(frame.frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel)
        # Find contours on the foreground
        detections = self._get_contours(fgmask)
        return detections

    def _get_contours(self, frame: MatLike) -> set[Detection]:
        """
        Finds contours in the mask and returns DetectionObject instances for each contour.

        Args:
            frame (MatLike): The binary mask image.

        Returns:
            List[DetectionObject]: List of detected objects from contours.
        """
        contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: set[Detection] = set()
        for _, contour in enumerate(contours):
            # Get the centroid of the object
            M = cv2.moments(contour)
            if M["m00"] != 0:  # Area is non-zero
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                logger.debug(f"Found object x: {cx} y: {cy}")
                x, y, w, h = cv2.boundingRect(contour)
                detections.add(Detection(Point(x, y), w, h))
                # cv2.drawContours(contour_frame, contour, -1, (0, 255, 0), 3)
        return detections
