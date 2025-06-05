import argparse
import logging
import os
import sys

import cv2
from dotenv import load_dotenv

from .constants import BATOMETER, VIDEO_FPS
from .frame import Frame
from .objectfinder import ObjectFinder
from .tracker import EuclideanDistTracker
from .utils import calculate_video_time_from_frame_num, load_video
from .window import ImageTransformer, resize_window_to_screen

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
    global VIDEO_FPS
    VIDEO_FPS = fps  # Dynamically set FPS for this run
    if not video.isOpened():
        logger.error(f"Failed to open video at {video_path}")
        sys.exit(1)
    objectFinder = ObjectFinder()
    # objectFinder.initialise(video)
    tracker = EuclideanDistTracker()
    img_transformer = ImageTransformer()

    cv2.namedWindow("main", cv2.WINDOW_NORMAL)

    # --- Play/Pause and Frame Cache Setup ---
    play = True
    frame_cache: list[
        tuple[
            "cv2.typing.MatLike",  # vid_frame
            int,  # frame_num
            str,  # frame_time
            Frame,  # Frame object
            set,  # detections (could be more specific if you have a Detection type)
            set,  # tracked_detections (could be more specific if you have a TrackedDetection type)
            "cv2.typing.MatLike",  # objectsFrame
        ]
    ] = []
    current_frame_idx = -1

    while True:
        if play:
            ret, vid_frame = video.read()
            frame_num = int(video.get(cv2.CAP_PROP_POS_FRAMES))
            if not ret:
                logger.error(f"Can't receive frame (stream end?). FrameNum: {frame_num}. Exiting ...")
                break
            frame_time = calculate_video_time_from_frame_num(frame_num, fps)
            frame = Frame(vid_frame, frame_num)
            logger.info(f"Opened frame: {frame_num}/{noFrames} - time: {frame_time}")

            # Find objects
            detections, objectsFrame = objectFinder.update(frame)
            for detection in detections:
                cv2.rectangle(
                    frame.frame,
                    (detection.point.x, detection.point.y),
                    (detection.point.x + detection.width, detection.point.y + detection.height),
                    (0, 0, 255),
                    3,
                )
            # Track objects
            tracked_detections = tracker.update(detections)
            # Draw tracked detections
            for obj in tracked_detections:
                color = ((obj.id * 70) % 256, (obj.id * 150) % 256, (obj.id * 230) % 256)
                overlay = frame.frame.copy()
                cv2.circle(
                    overlay,
                    (obj.predicted_position.x, obj.predicted_position.y),
                    radius=obj.prediction_range,
                    color=color,
                    thickness=-1,
                )
                alpha = 0.3  # Transparency factor (0.0 - fully transparent, 1.0 - opaque)
                cv2.addWeighted(overlay, alpha, frame.frame, 1 - alpha, 0, frame.frame)
                if hasattr(obj, "history") and len(obj.history) > 1:
                    last_valid_point = None
                    for i in range(1, len(obj.history)):
                        pt1 = obj.history[i - 1]
                        pt2 = obj.history[i]
                        if pt1 is not None and pt2 is not None:
                            cv2.line(frame.frame, (pt1.x, pt1.y), (pt2.x, pt2.y), color, 2)
                            last_valid_point = pt2
                        elif pt2 is not None and last_valid_point is not None:
                            cv2.line(
                                frame.frame,
                                (last_valid_point.x, last_valid_point.y),
                                (pt2.x, pt2.y),
                                color,
                                2,
                            )
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
            # Cache the frame and results
            frame_cache.append(
                (
                    vid_frame.copy(),
                    frame_num,
                    frame_time,
                    Frame(vid_frame.copy(), frame_num),
                    detections,
                    tracked_detections,
                    objectsFrame,
                )
            )
            current_frame_idx += 1
            # Use the current frame for display
            display_frame_width = frame.frame
        else:
            # Paused: show cached frame
            if 0 <= current_frame_idx < len(frame_cache):
                (
                    vid_frame,
                    frame_num,
                    frame_time,
                    cached_frame,
                    detections,
                    tracked_detections,
                    objectsFrame,
                ) = frame_cache[current_frame_idx]
                display_frame_width = cached_frame.frame
            else:
                # Nothing to show
                break
        # --- Draw play/pause button indicator ---
        status_text = "PAUSED" if not play else "PLAYING"
        overlay_text = (
            f"{status_text} | Space: Play/Pause | <-/->: Step | Esc: Quit | Frame: {current_frame_idx}"
        )
        # Draw overlay text at the top of the frame
        overlay_frame = display_frame_width.copy()
        cv2.rectangle(overlay_frame, (0, 0), (overlay_frame.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(
            overlay_frame,
            overlay_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        # Concatenate overlay_frame and objectsFrame side by side for display
        display_frame = img_transformer.images_side_by_side(overlay_frame, objectsFrame, "Frame", "Objects")
        cv2.imshow("main", display_frame)

        # Get width and height from combined_frame for resizing
        display_frame_height, display_frame_width = display_frame.shape[:2]
        resize_window_to_screen("main", display_frame_width, display_frame_height)

        key = cv2.waitKey(0 if not play else 30)
        # Handle arrow keys for your platform and Windows/Linux
        LEFT_KEYS = [2, 81]
        RIGHT_KEYS = [3, 83]
        if key == 27:  # ESC
            break
        elif key == 32:  # Spacebar
            play = not play
        elif key in LEFT_KEYS:
            if not play and current_frame_idx > 0:
                current_frame_idx -= 1
        elif key in RIGHT_KEYS:
            if not play and current_frame_idx < len(frame_cache) - 1:
                current_frame_idx += 1
            elif not play and current_frame_idx == len(frame_cache) - 1:
                # Step forward: play one frame
                play = True
        # else: ignore other keys
        if play and current_frame_idx < len(frame_cache) - 1:
            # If user resumes play from paused and is not at the end, jump to end
            current_frame_idx = len(frame_cache) - 1
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
