from dataclasses import dataclass

from cv2.typing import MatLike


@dataclass(frozen=True)
class Frame:
    frame: MatLike
    num: int
