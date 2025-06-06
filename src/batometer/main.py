import argparse
import cProfile
import logging
import os
import re
import sys

import cv2
import numpy as np
from dotenv import load_dotenv

from batometer.heatmap import Heatmap

from .constants import (
    AVERAGE_HEATMAP_KEYS,
    BATOMETER,
    ESC_KEYS,
    LEFT_KEYS,
    RIGHT_KEYS,
    SPACE_KEYS,
    TRACK_KEYS,
    VIDEO_FPS,
)
from .frameCache import FrameCacheEntry
from .objectfinder import ObjectFinder
from .tracker import EuclideanDistTracker
from .utils import calculate_video_time_from_frame_num, load_video
from .window import (
    ImageTransformer,
    OverlayMode,
    draw_detection_rectangle,
    draw_overlay_text,
    draw_predicted_object,
    draw_tracked_object,
    resize_window_to_screen,
)

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
    global VIDEO_FPS
    VIDEO_FPS = fps  # Dynamically set FPS for this run
    if not video.isOpened():
        logger.error(f"Failed to open video at {video_path}")
        sys.exit(1)
    objectFinder = ObjectFinder()
    # objectFinder.initialise(video)
    tracker = EuclideanDistTracker()
    img_transformer = ImageTransformer()
    heatmap = Heatmap(width, height)

    cv2.namedWindow("main", cv2.WINDOW_NORMAL)

    autoplay = True
    overlay_mode = OverlayMode.NONE
    frame_cache: list[FrameCacheEntry] = []
    current_paused_frame_idx = -1

    while True:
        if autoplay:
            ret, vid_frame = video.read()
            frame_num = int(video.get(cv2.CAP_PROP_POS_FRAMES))
            if not ret:
                logger.error(f"Can't receive frame (stream end?). FrameNum: {frame_num}. Exiting ...")
                break
            frame_time = calculate_video_time_from_frame_num(frame_num, fps)
            frame = Frame(vid_frame, frame_num)
            logger.info(f"Opened frame: {frame_num}/{noFrames} - time: {frame_time}")

            # Find objects
            detections, objects_frame = objectFinder.update(frame)
            for detection in detections:
                draw_detection_rectangle(frame.frame, detection)

            tracked_detections, predicted_objs = tracker.update(detections)
            for obj in predicted_objs:
                draw_predicted_object(frame.frame, obj)
            for obj in tracked_detections:
                draw_tracked_object(frame.frame, obj)

            heatmap.update(tracked_detections)
            # --- Generate heatmap for this frame ---
            long_tracks = set(
                obj
                for obj in tracker.current_potential_objects.union(tracker.all_objects)
                if hasattr(obj, "history") and len([pt for pt in obj.history if pt is not None]) > 10
            )
            idx = len(frame_cache)
            heat = np.zeros((height, width), dtype=np.float32)
            tracks_frame = np.zeros((height, width, 3), dtype=np.uint8)
            # --- Update direction grid for average heatmap ---
            for obj in long_tracks:
                for i in range(1, min(idx + 1, len(obj.history))):
                    pt1 = obj.history[i - 1]
                    pt2 = obj.history[i]
                    if pt1 is not None and pt2 is not None:
                        cv2.arrowedLine(
                            tracks_frame, (pt1.x, pt1.y), (pt2.x, pt2.y), (0, 255, 255), 2, tipLength=0.4
                        )
            # Replace norm_heat with a blue background (BGR: (255, 0, 0))
            blue_background = np.zeros((height, width, 3), dtype=np.uint8)
            blue_background[:, :] = (255, 0, 0)  # Blue in BGR
            color_heat = cv2.addWeighted(vid_frame, 0.1, tracks_frame, 1.0, 0)
            color_heat = cv2.addWeighted(color_heat, 1.0, vid_frame, 1.0, 0)

            heatmap_overlay = heatmap.add_overlay(frame.frame)
            # Cache the frame and results
            frame_cache.append(
                FrameCacheEntry(
                    vid_frame.copy(),
                    objects_frame.copy(),
                    color_heat.copy(),
                    heatmap_overlay.copy(),
                    frame_num,
                    frame_time,
                    detections,
                    tracked_detections,
                )
            )
            current_paused_frame_idx += 1
            video_frame = frame.frame
        else:
            # Paused: show cached frame
            frame_cache_entry = frame_cache[current_paused_frame_idx]
            video_frame = frame_cache_entry.video_frame
            objects_frame = frame_cache_entry.objects_frame
            frame_num = frame_cache_entry.frame_num
            frame_time = frame_cache_entry.frame_time
            detections = frame_cache_entry.detections
            tracked_detections = frame_cache_entry.tracked_detections

        # --- Draw play/pause button indicator ---
        video_frame = draw_overlay_text(video_frame, autoplay, current_paused_frame_idx)
        # --- Overlay heatmap if toggled ---
        match overlay_mode:
            case OverlayMode.TRACKS:
                video_overlay_frame = frame_cache[current_paused_frame_idx].tracks_frame
            case OverlayMode.HEATMAP:
                video_overlay_frame = frame_cache[current_paused_frame_idx].heatmap_frame
            case _:
                video_overlay_frame = video_frame

        display_frame = img_transformer.images_side_by_side(
            video_overlay_frame, objects_frame, "Frame", "Objects"
        )
        cv2.imshow("main", display_frame)

        # Only set window size once, after window creation and first frame
        if current_paused_frame_idx == 0:
            display_frame_height, display_frame_width = display_frame.shape[:2]
            resize_window_to_screen("main", display_frame_width, display_frame_height)

        key = cv2.waitKey(0 if not autoplay else 30)
        print(f"DEBUG: Key pressed: {key}")  # Add this line to debug key codes
        match key:
            case k if k in ESC_KEYS:
                break
            case k if k in SPACE_KEYS:
                autoplay = not autoplay
            case k if k in LEFT_KEYS:
                if not autoplay and current_paused_frame_idx > 0:
                    current_paused_frame_idx -= 1
                elif autoplay:
                    autoplay = not autoplay
                    current_paused_frame_idx -= 1
            case k if k in RIGHT_KEYS:
                if not autoplay:
                    if current_paused_frame_idx < len(frame_cache) - 1:
                        current_paused_frame_idx += 1
                    elif current_paused_frame_idx == len(frame_cache) - 1:
                        autoplay = True
            case k if k in TRACK_KEYS:
                if overlay_mode != OverlayMode.TRACKS:
                    overlay_mode = OverlayMode.TRACKS
                else:
                    overlay_mode = OverlayMode.NONE
            case k if k in AVERAGE_HEATMAP_KEYS:
                if overlay_mode != OverlayMode.HEATMAP:
                    overlay_mode = OverlayMode.HEATMAP
                else:
                    overlay_mode = OverlayMode.NONE
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
