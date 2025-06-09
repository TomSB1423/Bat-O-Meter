from dataclasses import dataclass
from typing import List, Optional


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

    def __hash__(self) -> int:
        """Hash based on x and y coordinates."""
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

    def __hash__(self) -> int:
        """Hash based on point, width, and height."""
        return hash((self.point, self.width, self.height))


@dataclass
class IdentifiedObject(Detection):
    """
    Represents an identified object with an assigned unique identifier.

    Attributes:
        id (int): Unique identifier for the detected object.
        history (List[Optional[Point]]): List of previous positions (None if missed).
        speed (tuple[float, float]): (vx, vy) speed vector.
        predicted_position (Point): Predicted next position.
        prediction_range (int): Range for prediction.
        missed_tracks (int): Number of missed frames.
    """

    id: int
    history: List[Optional[Point]]
    speed: tuple[float, float]  # (vx, vy)
    predicted_position: Point
    prediction_range: int = 30
    missed_tracks: int = 0

    def __init__(self, id: int, detectionObject: Detection) -> None:
        """
        Initialize an IdentifiedObject from a Detection and an id.

        Args:
            id (int): Unique identifier for the object.
            detectionObject (Detection): The detected object to copy bounding box from.
        """
        super().__init__(detectionObject.point, detectionObject.width, detectionObject.height)
        self.id = id
        self.history = [detectionObject.point]
        self.predicted_position = detectionObject.point
        self.speed = (0.0, 0.0)

    def __hash__(self) -> int:
        """Hash based on the unique id."""
        return hash(self.id)

    def update(self, point: Optional[Point], width=0, height=0) -> None:
        """
        Update the object's location and prediction based on a new point.
        If point is None, increment missed_tracks and update prediction.
        If point is provided, update speed, prediction, and reset missed_tracks.

        Args:
            point (Optional[Point]): The new detected point or None if missed.
        """
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
        last_point: Optional[Point] = None
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
        self.width = width
        self.height = height

    def is_self(self, det: Detection) -> bool:
        """
        Determine if a detection is inside the predicted circle for this object.

        Args:
            det (Detection): The detection to compare.

        Returns:
            bool: True if the detection's point is inside the predicted circle, False otherwise.
        """
        dx = det.point.x - self.predicted_position.x
        dy = det.point.y - self.predicted_position.y
        return dx * dx + dy * dy <= self.prediction_range * self.prediction_range
