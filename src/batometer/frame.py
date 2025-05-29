from dataclasses import dataclass

from cv2.typing import MatLike


@dataclass(frozen=True)
class Frame:
    """
    Represents a single video frame with its associated frame number.

    Attributes:
        frame (MatLike): The image data for the frame.
        num (int): The frame number in the video sequence.
    """

    frame: MatLike
    num: int
