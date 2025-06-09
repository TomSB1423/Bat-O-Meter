import logging
import os
import shutil
from pathlib import Path

import cv2
from cv2.typing import MatLike

from .constants import BATOMETER

logger = logging.getLogger(f"{BATOMETER}.utils")

TEMP_DIR = os.path.join(Path(__file__).parent.parent, ".temp")
if os.path.exists(TEMP_DIR):
    shutil.rmtree(TEMP_DIR)


def save_image_to_temp(img: MatLike, frame_num: int) -> None:
    """
    Saves an image to the temporary directory with the frame number in the filename.

    Args:
        img (MatLike): The image to save.
        frame_num (int): The frame number for the filename.
    Raises:
        Exception: If the image could not be saved.
    """
    os.makedirs(TEMP_DIR, exist_ok=True)
    path = os.path.join(TEMP_DIR, f"frame-{frame_num}.png")
    if not cv2.imwrite(path, img):
        raise Exception("Could not save image")
    logger.debug(f"Saved frame no.{frame_num} to {path}")


def images_to_mp4(folder_path: str, output_video_path: str, fps: int) -> None:
    """
    Converts a sequence of images in a folder to an MP4 video.

    Args:
        folder_path (str): Path to the folder containing images.
        output_video_path (str): Path to save the output video.
        fps (int): Frames per second for the output video.
    """
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
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")  # Codec for MP4
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Add each image to the video
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        frame = cv2.imread(image_path)
        video_writer.write(frame)

    # Release the video writer
    video_writer.release()
    print(f"Video saved as {output_video_path}")
