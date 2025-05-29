import logging
import math
from typing import List

from .constants import BATOMETER
from .detectionObject import Detection, IdentifiedObject, Point

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
        self.all_tracks: List[IdentifiedObject] = []
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count: int = 0

    def update(self, detected_objects: List[Detection]) -> List[IdentifiedObject]:
        """
        Updates the tracker with new detections, assigns IDs, and returns identified objects.

        Args:
            detected_objects (List[DetectionObject]): List of detected objects in the current frame.

        Returns:
            List[IdentifiedObject]: List of objects with assigned unique IDs.
        """
        # Objects boxes and ids
        currently_tracked: List[IdentifiedObject] = []

        # Get center point of new object
        for obj in detected_objects:
            # Find out if that object was detected already
            for detection in self.all_tracks:
                dist = math.hypot(obj.x - detection.x, obj.y - detection.y)

                if dist < 200:
                    detection.update_location(Point(obj.x, obj.y))
                    currently_tracked.append(detection)
                    break

            # New object is detected we assign the ID to that object
            new_identified_object = IdentifiedObject(self.id_count, obj)
            self.all_tracks.append(new_identified_object)
            currently_tracked.append(new_identified_object)
            self.id_count += 1
       
        return currently_tracked
