import logging
import os
import sys
from contextlib import AbstractContextManager
from pathlib import Path

import cv2

from .constants import BATOMETER

FORMAT = "%(asctime)s %(levelname)-8s %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(BATOMETER)


class VideoManager(AbstractContextManager):
    def __init__(self, video_path):
        self.video, self.width, self.height, self.fps, self.max_frames, self.frame_time = self._load_video(
            video_path
        )
        self.video_path = Path(video_path)
        if not self.video.isOpened():
            logger.error(f"Failed to open video at {video_path}")
            sys.exit(1)
        self.frame_num = 0

    def _load_video(self, path_str: str) -> tuple["cv2.VideoCapture", int, int, int, int, str]:
        """
        Loads a video from the given path and returns the video capture object and its properties.

        Args:
            path_str (str): Path to the video file.

        Returns:
            tuple[cv2.VideoCapture, int, int, int, int]:
                - video (cv2.VideoCapture): The video capture object.
                - width (int): Frame width.
                - height (int): Frame height.
                - fps (int): Frames per second.
                - noFrames (int): Total number of frames.
        Raises:
            FileNotFoundError: If the video file does not exist.
        """
        path = Path(path_str)
        if not os.path.isfile(path):
            logger.error(f"Video file does not exist: {path}")
            raise FileNotFoundError(f"Video file does not exist: {path}")
        video = cv2.VideoCapture(str(path))
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video.get(cv2.CAP_PROP_FPS))
        noFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_time = self._calculate_video_time_from_frame_num(0, fps)
        logger.info(
            f"Video information // Width: {width} - Height: {height} - FPS: {fps} - numFrames: {noFrames} - "
            f"Length: {frame_time}"
        )
        return video, width, height, fps, noFrames, frame_time

    def _calculate_video_time_from_frame_num(self, frame_num: int, fps: int) -> str:
        """
        Calculates the video time string from a frame number and fps.

        Args:
            frame_num (int): The frame number.
            fps (int): Frames per second.

        Returns:
            str: Time in HH:MM:SS format.
        """
        hours = int(frame_num / (fps * 3600))
        minutes = int(frame_num / (fps * 60) % 60)
        seconds = int((frame_num / fps) % 60)
        milli_seconds = int(frame_num % fps)
        time = "{0:02d}:{1:02d}:{2:02d}:{3:02d}".format(hours, minutes, seconds, milli_seconds)
        return time

    def read_frame(self):
        ret, frame = self.video.read()
        if not ret:
            raise Exception("Can't receive frame (stream end?). FrameNum: {self.frame_num}. Exiting ...")
        self.frame_num += 1
        self.frame_time = self._calculate_video_time_from_frame_num(self.frame_num, self.fps)
        return frame

    def has_more_frames(self) -> bool:
        """
        Checks if there are more frames to read in the video.

        Returns:
            bool: True if there are more frames to read, False otherwise.
        """
        return self.frame_num < self.max_frames

    def release(self):
        self.video.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
