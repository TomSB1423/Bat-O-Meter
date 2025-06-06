from dataclasses import dataclass
from typing import Any

import cv2

from .detectionObject import Detection, IdentifiedObject
from .frame import Frame


@dataclass
class FrameCacheEntry:
    video_frame: "cv2.typing.MatLike"
    objects_frame: "cv2.typing.MatLike"
    tracks_frame: "cv2.typing.MatLike"
    heatmap_frame: "cv2.typing.MatLike"
    frame_num: int
    frame_time: str
    detections: set[Detection]
    tracked_detections: set[IdentifiedObject]
