import cv2
import logging

logger = logging.getLogger("Bat-O-Meter.utils")


def load_video(path):
    """Loads the video"""
    video = cv2.VideoCapture(path)
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = video.get(cv2.CAP_PROP_FPS)
    noFrames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    logger.info(
        f"Video information // Width: {width} - Height: {height} - FPS: {fps} - numFrames: {noFrames} - Length: {caluclate_time(noFrames, fps)}"
    )
    return video, width, height, fps


def caluclate_time(frame_no, fps) -> str:
    hours = int(frame_no / (fps * 3600))
    minutes = int(frame_no / (fps * 60) % 60)
    seconds = int((frame_no / fps) % 60)
    milli_seconds = int(frame_no % fps)
    time = "{0:02d}:{1:02d}:{2:02d}:{3:02d}".format(hours, minutes, seconds, milli_seconds)
    return time
