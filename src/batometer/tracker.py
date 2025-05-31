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
        self.all_tracks: set[IdentifiedObject] = set()
        self.current_objects: set[IdentifiedObject] = set()
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count: int = 0

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
        # Remove tracks that have not been seen for more than MAX_MISSED_FRAMES
        self.current_objects = set(obj for obj in self.current_objects if obj.missed_tracks < MAX_MISSED_FRAMES)
        # First iterate through already identified objects
        # match these to any detections using object predicted location
        # then if detections still not found
        # find objects that are max 10 frames behind, using their predicted location
        # if still not found then assume a new object
        successfully_matched = set()
        for det in detected_objects:
            matched = False
            for track in self.current_objects:
                dist = math.hypot(det.point.x - track.predicted_position.x, det.point.y - track.predicted_position.y)
                if dist < 20 * (track.missed_tracks + 1):
                    track.update_location(det.point)
                    matched = True
                    successfully_matched.add(track.id)
                    break
            if not matched:
                new_obj = IdentifiedObject(self.id_count, det)
                self.id_count += 1
                self.all_tracks.add(new_obj)
                self.current_objects.add(new_obj)
                self.all_tracks.add(new_obj)
        for track in self.current_objects:
            if track.id not in successfully_matched:
                track.update_location(None)
        return self.current_objects
