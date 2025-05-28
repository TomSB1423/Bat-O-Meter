import logging
import os
from typing import Tuple
from pathlib import Path

import cv2
from cv2.typing import MatLike

from constants import *

logger = logging.getLogger(f"{BATOMETER}.utils")


def load_video(path_str: str) -> Tuple[cv2.VideoCapture, int, int, int, int]:
    """Loads the video"""
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
    logger.info(
        f"Video information // Width: {width} - Height: {height} - FPS: {fps} - numFrames: {noFrames} - Length: {caluclate_video_time_from_frame_num(noFrames, fps)}"
    )
    return video, width, height, fps, noFrames


def caluclate_video_time_from_frame_num(frame_num: int, fps: int) -> str:
    hours = int(frame_num / (fps * 3600))
    minutes = int(frame_num / (fps * 60) % 60)
    seconds = int((frame_num / fps) % 60)
    milli_seconds = int(frame_num % fps)
    time = "{0:02d}:{1:02d}:{2:02d}:{3:02d}".format(hours, minutes, seconds, milli_seconds)
    return time


def save_image_to_temp(img: MatLike, frame_num: int) -> None:
    cv2.imwrite(
        str(
            os.path.join(
                Path(__file__).parent.parent,
                ".temp",
                f"frame-{frame_num}.png",
            )
        ),
        img,
    )


def images_to_mp4(folder_path: str, output_video_path: str, fps: int) -> None:
    # Get all image files from the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith((".png", ".jpg", ".jpeg"))]

    # Sort files if needed (e.g., by name)
    image_files.sort(key=lambda f: int("".join(filter(str.isdigit, f))))

    # Check if there are any images
    if not image_files:
        print("No images found in the specified folder.")
        exit()

    # Read the first image to get dimensions
    first_image_path = os.path.join(folder_path, image_files[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Add each image to the video
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        frame = cv2.imread(image_path)
        video_writer.write(frame)

    # Release the video writer
    video_writer.release()
    print(f"Video saved as {output_video_path}")
