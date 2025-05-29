import argparse
import logging
import math
import os
import sys

import cv2
from dotenv import load_dotenv

from constants import BATOMETER
from detectionObject import DetectionObject, IdentifiedObject
from frame import Frame
from objectfinder import ObjectFinder
from tracker import EuclideanDistTracker
from utils import calculate_video_time_from_frame_num, load_video
from window import ImageTransformer

FORMAT = "%(asctime)s %(levelname)-8s %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(BATOMETER)


def main(video_path: str) -> None:
    """
    Main entry point for Bat-O-Meter. Loads a video, detects and tracks objects frame by frame.

    Args:
        video_path (str): Path to the video file to process.
    """
    video, width, height, fps, noFrames = load_video(video_path)
    if not video.isOpened():
        logger.error(f"Failed to open video at {video_path}")
        sys.exit(1)
    objectFinder = ObjectFinder()
    objectFinder.initialise(video)
    tracker = EuclideanDistTracker()
    img_transformer = ImageTransformer()

    while video.isOpened():
        ret, vid_frame = video.read()
        frame_num = int(video.get(cv2.CAP_PROP_POS_FRAMES))
        if not ret:
            logger.error(f"Can't receive frame (stream end?). FrameNum: {frame_num}. Exiting ...")
            break
        frame_time = calculate_video_time_from_frame_num(frame_num, fps)
        frame = Frame(vid_frame, frame_num)
        logger.info(f"Opened frame: {frame_num}/{noFrames} - time: {frame_time}")

        # Find objects
        detections: list[DetectionObject] = objectFinder.update(frame)
        for detection in detections:
            cv2.rectangle(
                frame.frame,
                (detection.x, detection.y),
                (detection.x + detection.width, detection.y + detection.height),
                (0, 0, 255),
                3,
            )

        # Clean bogus detections by looking in the future
        next_frame = frame_num + 1
        video.set(cv2.CAP_PROP_POS_FRAMES, next_frame)
        _, fut_frame = video.read()
        future_detections: list[DetectionObject] = objectFinder.update(Frame(fut_frame, next_frame))
        cleaned_detections: list[DetectionObject] = []
        for detection in detections:
            for fut_detection in future_detections:
                dist = math.hypot(detection.x - fut_detection.x, detection.y - fut_detection.y)
                if dist < 100:
                    cleaned_detections.append(detection)
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

        # Track objects
        tracked_detections: list[IdentifiedObject] = tracker.update(cleaned_detections)

        # Display tracked detections
        for detection in tracked_detections:
            cv2.putText(
                frame.frame,
                str(detection.id),
                (detection.x, detection.y - 15),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (255, 0, 0),
                2,
            )
            cv2.rectangle(
                frame.frame,
                (detection.x, detection.y),
                (detection.x + detection.width, detection.y + detection.height),
                (0, 255, 0),
                3,
            )
        cv2.imshow("main", frame.frame)
        key = cv2.waitKey(30)
        if key == 27:
            break
        # save_image_to_temp(frame.frame, frame_num)

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    """
    Command-line interface entry point for Bat-O-Meter. Parses arguments and starts main processing.
    """
    load_dotenv()
    parser = argparse.ArgumentParser(description="Bat-O-Meter video object detection and tracking")
    parser.add_argument(
        "--video-path",
        type=str,
        default=os.getenv("VIDEO_PATH"),
        help="Path to the video file (or set VIDEO_PATH env variable)",
    )
    args = parser.parse_args()
    if not args.video_path:
        logger.error("No video path provided. Use --video-path or set VIDEO_PATH in .env.")
        sys.exit(1)
    main(args.video_path)
