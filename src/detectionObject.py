from dataclasses import dataclass


@dataclass
class Point:
    x: int
    y: int


@dataclass
class DetectionObject:
    """
    Represents a detected object in a video frame.

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

    @property
    def center_point(self) -> Point:
        """
        Returns the center (x, y) of the bounding box as a named tuple.
        """
        return Point((self.x + self.x + self.width) // 2, (self.y + self.y + self.height) // 2)


@dataclass
class IdentifiedObject(DetectionObject):
    """
    Represents a detected object with an assigned unique identifier.

    Attributes:
        id (int): Unique identifier for the detected object.
        (Other attributes inherited from DetectionObject)
    """

    id: int

    def __init__(self, id: int, detectionObject: DetectionObject):
        super().__init__(detectionObject.x, detectionObject.y, detectionObject.width, detectionObject.height)
        self.id = id
