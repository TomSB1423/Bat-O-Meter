import logging
import math
from typing import Dict, List

from constants import *
from detectionObject import DetectionObject, IdentifiedObject, Point

logger = logging.getLogger(f"{BATOMETER}.EuclideanDistTracker")


class EuclideanDistTracker:
    """
    Tracks objects across video frames using Euclidean distance between their center points.
    Assigns unique IDs to detected objects and maintains their identities across frames.
    """

    def __init__(self) -> None:
        """
        Initializes the tracker with an empty dictionary for center points and an ID counter.
        """
        # Store the center positions of the objects
        self.center_points: Dict[int, Point] = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count: int = 0

    def update(self, detected_objects: List[DetectionObject]) -> List[IdentifiedObject]:
        """
        Updates the tracker with new detections, assigns IDs, and returns identified objects.

        Args:
            detected_objects (List[DetectionObject]): List of detected objects in the current frame.

        Returns:
            List[IdentifiedObject]: List of objects with assigned unique IDs.
        """
        # Objects boxes and ids
        identified_objects: List[IdentifiedObject] = []

        # Get center point of new object
        for obj in detected_objects:
            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(obj.center_point.x - pt.x, obj.center_point.y - pt.y)

                if dist < 50:
                    self.center_points[id] = obj.center_point
                    print(self.center_points)
                    identified_objects.append(IdentifiedObject(id, obj))
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if not same_object_detected:
                self.center_points[self.id_count] = obj.center_point
                identified_objects.append(IdentifiedObject(self.id_count, obj))
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points: Dict[int, Point] = {}
        for obj in identified_objects:
            center = self.center_points[obj.id]
            new_center_points[obj.id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return identified_objects
