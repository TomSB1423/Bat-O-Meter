from typing import List

from dataclasses import dataclass


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


@dataclass
class Detection:
    """
    Represents a detection in a video frame.

    Attributes:
        x (int): X-coordinate of the top-left corner of the bounding box.
        y (int): Y-coordinate of the top-left corner of the bounding box.
        width (int): Width of the bounding box.
        height (int): Height of the bounding box.
    """

    x: int
    y: int
    width: int
    height: int


@dataclass
class IdentifiedObject(Detection):
    """
    Represents an identified object with an assigned unique identifier.

    Attributes:
        id (int): Unique identifier for the detected object.
        (Other attributes inherited from DetectionObject)
    """

    id: int
    history: List[Point]

    def __init__(self, id: int, detectionObject: Detection) -> None:
        """
        Initialize an IdentifiedObject from a DetectionObject and an id.

        Args:
            id (int): Unique identifier for the object.
            detectionObject (DetectionObject): The detected object to copy bounding box from.
        """
        super().__init__(detectionObject.x, detectionObject.y, detectionObject.width, detectionObject.height)
        self.id = id
        self.history = [Point(detectionObject.x, detectionObject.y)]

    def update_location(self, point: Point) -> None:
        self.history.append(point)
