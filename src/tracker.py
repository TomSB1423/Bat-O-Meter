import logging
import math

from constants import *
from detectionObject import DetectionObject, IdentifiedObject, Point

logger = logging.getLogger(f"{BATOMETER}.EuclideanDistTracker")


class EuclideanDistTracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points: dict[int, Point] = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0

    def update(self, detected_objects: list[DetectionObject]) -> list[IdentifiedObject]:
        # Objects boxes and ids
        identified_objects: list[IdentifiedObject] = []

        # Get center point of new object
        for object in detected_objects:
            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(object.center_point.x - pt.x, object.center_point.y - pt.y)

                if dist < 50:
                    self.center_points[id] = object.center_point
                    print(self.center_points)
                    identified_objects.append(IdentifiedObject(id, object))
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = object.center_point
                identified_objects.append(IdentifiedObject(self.id_count, object))
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for object in identified_objects:
            center = self.center_points[object.id]
            new_center_points[object.id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return identified_objects
