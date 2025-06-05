import math
from dataclasses import dataclass
from typing import List


@dataclass
class Point:
    """
    Represents a 2D point in image coordinates.

    Attributes:
        x (int): X-coordinate.
        y (int): Y-coordinate.
    """

    x: int
    y: int

    def __hash__(self):
        return hash((self.x, self.y))


@dataclass
class Detection:
    """
    Represents a detection in a video frame.

    Attributes:
        point (Point): X and Y coordinate of the top-left corner of the bounding box.
        width (int): Width of the bounding box.
        height (int): Height of the bounding box.
    """

    point: Point
    width: int
    height: int

    def __hash__(self):
        return hash((self.point, self.width, self.height))


@dataclass
class IdentifiedObject(Detection):
    """
    Represents an identified object with an assigned unique identifier.

    Attributes:
        id (int): Unique identifier for the detected object.
        (Other attributes inherited from DetectionObject)
    """

    id: int
    history: List[Point | None]
    speed: tuple[float, float]  # (vx, vy)
    predicted_position: Point
    prediction_range: int = 30
    missed_tracks: int = 0

    def __init__(self, id: int, detectionObject: Detection) -> None:
        """
        Initialize an IdentifiedObject from a DetectionObject and an id.

        Args:
            id (int): Unique identifier for the object.
            detectionObject (DetectionObject): The detected object to copy bounding box from.
        """
        super().__init__(detectionObject.point, detectionObject.width, detectionObject.height)
        self.id = id
        self.history = [detectionObject.point]
        self.predicted_position = detectionObject.point
        self.speed = (0.0, 0.0)

    def __hash__(self):
        return hash(self.id)

    def update_location(self, point: Point | None) -> None:
        if point is None:
            self.missed_tracks += 1
            predicted_x = self.point.x + (self.speed[0] * self.missed_tracks)
            predicted_y = self.point.y + (self.speed[1] * self.missed_tracks)
            self.predicted_position = Point(int(predicted_x), int(predicted_y))
            self.history.append(None)
            # Do not update speed if no new point
            return
        self.missed_tracks = 0
        # Find last non-None point for speed calculation
        last_point = None
        passed_frames = 0
        for prev in reversed(self.history):
            passed_frames += 1
            if prev is not None:
                last_point = prev
                break
        if last_point is not None:
            dx = (point.x - last_point.x) / passed_frames
            dy = (point.y - last_point.y) / passed_frames
            self.speed = (dx, dy)
        self.point = point
        predicted_x = self.point.x + self.speed[0]
        predicted_y = self.point.y + self.speed[1]
        self.predicted_position = Point(int(predicted_x), int(predicted_y))
        self.history.append(point)

    def is_self(self, det: Detection) -> bool:
        if self.speed == 0:
            dist = math.hypot(det.point.x - self.point.x, det.point.y - self.point.y)
            if dist < self.prediction_range:
                return True
        dist = math.hypot(det.point.x - self.predicted_position.x, det.point.y - self.predicted_position.y)
        if dist < self.prediction_range * (self.missed_tracks + 1):
            return True
        return False
