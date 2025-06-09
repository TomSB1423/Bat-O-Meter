import logging

import cv2
import numpy as np

from .constants import BATOMETER
from .detectionObject import Detection, IdentifiedObject

logger = logging.getLogger(f"{BATOMETER}.ObjectTracker")


class ObjectTracker:
    """
    Tracks objects across video frames using Euclidean distance between their center points.
    Assigns unique IDs to detected objects and maintains their identities across frames.
    """

    all_objects: set["IdentifiedObject"]
    current_potential_objects: set["IdentifiedObject"]
    id_count: int

    def __init__(self, width: int, height: int) -> None:
        """
        Initializes the EuclideanDistTracker.
        Sets up storage for all tracked objects, currently tracked objects, and the ID counter.
        """
        # Store the center positions of the objects
        self.width = width
        self.height = height
        self.all_objects: set[IdentifiedObject] = set()
        self.current_potential_objects: set[IdentifiedObject] = set()
        self.pixel_heatmap = np.zeros((self.height, self.width), dtype=np.float32)
        self.max_missed_frames = 10
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count: int = 0

    def update(
        self, detected_objects: set["Detection"]
    ) -> tuple[set["IdentifiedObject"], set["IdentifiedObject"]]:
        """
        Updates the tracker with new detections, assigns IDs, and returns identified objects.

        Args:
            detected_objects (set[Detection]): Set of detected objects in the current frame.

        Returns:
            set[IdentifiedObject]: Set of objects with assigned unique IDs for the current frame.
        """
        for obj in list(self.current_potential_objects):
            self.update_heatmap(obj)
            if obj.missed_tracks > self.max_missed_frames:
                self.current_potential_objects.remove(obj)
        current_objects: set[IdentifiedObject] = set()
        for obj in self.current_potential_objects:
            matched = False
            for det in detected_objects:
                if obj.is_self(det):
                    obj.update(det.point, det.width, det.height)
                    detected_objects.remove(det)
                    current_objects.add(obj)
                    matched = True
                    break
            if not matched:
                obj.update(None)
        for det in detected_objects:
            new_obj = IdentifiedObject(self.id_count, det)
            self.current_potential_objects.add(new_obj)
            current_objects.add(new_obj)
            self.all_objects.add(new_obj)
            self.id_count += 1

        return current_objects.copy(), self.current_potential_objects.difference(current_objects)

    def update_heatmap(self, obj: IdentifiedObject):
        left_p = 0
        tracks_frame = np.zeros_like(self.pixel_heatmap, dtype=np.uint8)
        while left_p < len(obj.history) - 1:
            pt1 = obj.history[left_p]
            # Move the second pointer to the next non-None point
            right_p = left_p + 1
            while right_p < len(obj.history) and obj.history[right_p] is None:
                right_p += 1

            if right_p < len(obj.history):
                pt2 = obj.history[right_p]
                if pt1 is not None and pt2 is not None:
                    cv2.line(tracks_frame, (pt1.x, pt1.y), (pt2.x, pt2.y), color=(1,), thickness=2)
            # Move the first pointer to the second pointer's position
            left_p = right_p
        self.pixel_heatmap += tracks_frame

    def create_overlay(self, frame):
        long_tracks = set(obj for obj in self.current_potential_objects.union(self.all_objects))
        tracks_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for obj in long_tracks:
            left_p = 0
            while left_p < len(obj.history) - 1:
                pt1 = obj.history[left_p]
                right_p = left_p + 1
                while right_p < len(obj.history) and obj.history[right_p] is None:
                    right_p += 1

                if right_p < len(obj.history):
                    pt2 = obj.history[right_p]
                    if pt1 is not None and pt2 is not None:
                        arrow_length = np.sqrt((pt2.x - pt1.x) ** 2 + (pt2.y - pt1.y) ** 2)
                        fixed_tip_length = min(10 / arrow_length, 0.2)  # Try to keep tip size consistent
                        cv2.arrowedLine(
                            tracks_frame,
                            (pt1.x, pt1.y),
                            (pt2.x, pt2.y),
                            (0, 255, 255),
                            2,
                            tipLength=fixed_tip_length,
                        )
                left_p = right_p

        blue_background = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        blue_background[:, :] = (255, 0, 0)  # Blue in BGR
        overlay = cv2.addWeighted(frame, 0.1, tracks_frame, 1.0, 0)
        overlay = cv2.addWeighted(overlay, 1.0, frame, 1.0, 0)
        return overlay

    def create_heatmap_overlay(self, frame):
        log_heatmap = self.pixel_heatmap
        heatmap_prob = np.zeros_like(log_heatmap, dtype=np.uint8)
        cv2.normalize(log_heatmap, heatmap_prob, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmap_img = cv2.applyColorMap(heatmap_prob, cv2.COLORMAP_JET)
        resized_heatmap_img = cv2.resize(heatmap_img, (self.width, self.height))
        final_heatmap_img = cv2.addWeighted(frame, 0.7, resized_heatmap_img, 0.3, 0)
        return final_heatmap_img

    # Add a method to calculate bat likelihood based on movement patterns
    def calculate_bat_likelihood(self, movement_pattern):
        """
        Estimate the likelihood of an object being a bat based on its movement pattern.

        Args:
            movement_pattern (list[tuple[int, int]]): List of (dx, dy) movements.

        Returns:
            float: Likelihood score (0.0 to 1.0).
        """
        # Example heuristic: bats tend to have erratic, fast movements
        total_distance = sum((dx**2 + dy**2) ** 0.5 for dx, dy in movement_pattern)
        avg_distance = total_distance / len(movement_pattern) if movement_pattern else 0

        # Normalize likelihood (example: >10 pixels per frame is likely a bat)
        likelihood = min(1.0, avg_distance / 10.0)
        return likelihood
