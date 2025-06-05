import logging

from .constants import BATOMETER
from .detectionObject import Detection, IdentifiedObject

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
        self.all_objects: set[IdentifiedObject] = set()
        self.current_potential_objects: set[IdentifiedObject] = set()
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count: int = 1

    def update(self, detected_objects: set[Detection]) -> set[IdentifiedObject]:
        """
        Updates the tracker with new detections, assigns IDs, and returns identified objects.

        Args:
            detected_objects (List[Detection]): List of detected objects in the current frame.
            frame_num (int): The current frame number (for tracking missed frames).

        Returns:
            List[IdentifiedObject]: List of objects with assigned unique IDs.
        """
        MAX_MISSED_FRAMES = 10
        self.current_potential_objects = set(
            obj for obj in self.current_potential_objects if obj.missed_tracks < MAX_MISSED_FRAMES
        )
        current_objects: set[IdentifiedObject] = set()
        for obj in self.current_potential_objects:
            matched = False
            for det in detected_objects:
                if obj.is_self(det):
                    obj.update_location(det.point)
                    detected_objects.remove(det)
                    current_objects.add(obj)
                    matched = True
                    break
            if not matched:
                obj.update_location(None)
        for det in detected_objects:
            new_obj = IdentifiedObject(self.id_count, det)
            self.current_potential_objects.add(new_obj)
            current_objects.add(new_obj)
            self.all_objects.add(new_obj)
            self.id_count += 1
        return current_objects
