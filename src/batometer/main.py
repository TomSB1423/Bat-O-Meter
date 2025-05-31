import argparse
import logging
import math
import os
import sys

import cv2
from dotenv import load_dotenv

from .constants import BATOMETER
from .frame import Frame
from .objectfinder import ObjectFinder
from .tracker import EuclideanDistTracker
from .utils import calculate_video_time_from_frame_num, load_video
from .window import ImageTransformer

FORMAT = "%(asctime)s %(levelname)-8s %(message)s"
logging.basicConfig(level=logging.DEBUG, format=FORMAT)
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
    # objectFinder.initialise(video)
    tracker = EuclideanDistTracker()
    img_transformer = ImageTransformer()

    cv2.namedWindow('main', cv2.WINDOW_NORMAL)
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
        detections = objectFinder.update(frame)
        for detection in detections:
            cv2.rectangle(
                frame.frame,
                (detection.point.x, detection.point.y),
                (detection.point.x + detection.width, detection.point.y + detection.height),
                (0, 0, 255),
                3,
            )

        # Clean bogus detections by looking in the future
        # This is a silly way of doing it as real objects might be there 
        # next_frame = frame_num + 1
        # video.set(cv2.CAP_PROP_POS_FRAMES, next_frame)
        # _, fut_frame = video.read()
        # max_movement_dist = 100
        # future_detections= objectFinder.update(Frame(fut_frame, next_frame))
        # cleaned_detections = set()
        # for detection in detections:
        #     for fut_detection in future_detections:
        #         movement_dist = math.hypot(detection.point.x - fut_detection.point.x, detection.point.y - fut_detection.point.y)
        #         if movement_dist < max_movement_dist:
        #             cleaned_detections.add(detection)
        # video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

        # Track objects
        tracked_detections = tracker.update(detections)

        # Display tracked detections
        for obj in tracked_detections:
            # Draw the track history as a line
            color = ((obj.id * 70) % 256, (obj.id * 150) % 256, (obj.id * 230) % 256)
            if hasattr(obj, "history") and len(obj.history) > 1:
                last_valid_point = None
                for i in range(1, len(obj.history)):
                    pt1 = obj.history[i - 1]
                    pt2 = obj.history[i]
                    if pt1 is not None and pt2 is not None:
                        cv2.line(frame.frame, (pt1.x, pt1.y), (pt2.x, pt2.y), color, 2)
                        last_valid_point = pt2
                    elif pt2 is not None and last_valid_point is not None:
                        cv2.line(frame.frame, (last_valid_point.x, last_valid_point.y), (pt2.x, pt2.y), color, 2)
                        last_valid_point = pt2
                    elif pt2 is not None:
                        last_valid_point = pt2
            cv2.putText(
                frame.frame,
                str(obj.id),
                (obj.point.x, obj.point.y - 15),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                color,
                2,
            )
            cv2.rectangle(
                frame.frame,
                (obj.point.x, obj.point.y),
                (obj.point.x + obj.width, obj.point.y + obj.height),
                color,
                3,
            )
        cv2.imshow("main", frame.frame)
        cv2.resizeWindow('main', int(width * 0.8), int(height * 0.8))
        
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
