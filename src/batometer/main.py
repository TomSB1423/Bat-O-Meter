import argparse
import logging
import os
import sys

import cv2
import numpy as np
from dotenv import load_dotenv

from .constants import BATOMETER, VIDEO_FPS
from .frame import Frame
from .objectfinder import ObjectFinder
from .tracker import EuclideanDistTracker
from .utils import calculate_video_time_from_frame_num, load_video
from .window import ImageTransformer, resize_window_to_screen

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

    cv2.namedWindow("main", cv2.WINDOW_NORMAL)

    # --- Play/Pause and Frame Cache Setup ---
    play = True
    show_heatmap = False  # Toggle for heatmap overlay
    show_average_heatmap = False
    average_heatmap = None
    avg_heat = np.zeros((height, width), dtype=np.float32)  # Persistent average heatmap
    frame_cache: list[
        tuple[
            "cv2.typing.MatLike",  # vid_frame
            int,  # frame_num
            str,  # frame_time
            Frame,  # Frame object
            set,  # detections
            set,  # tracked_detections
            "cv2.typing.MatLike",  # objectsFrame
            "cv2.typing.MatLike",  # heatmap for this frame
        ]
    ] = []
    current_frame_idx = -1

    # --- Directional grid for average heatmap ---
    grid_size = 32  # You can adjust this for finer/coarser arrows
    grid_h = height // grid_size
    grid_w = width // grid_size
    direction_sum_grid = np.zeros((grid_h, grid_w, 2), dtype=np.float32)  # (sum_dx, sum_dy) per cell
    direction_count_grid = np.zeros((grid_h, grid_w), dtype=np.int32)     # count per cell

    # --- Main loop ---
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
            tracked_detections, predicted_objs = tracker.update(detections)
            # Draw predictions
            for obj in predicted_objs:
                color = ((obj.id * 70) % 256, (obj.id * 150) % 256, (obj.id * 230) % 256)
                cv2.circle(
                    frame.frame,
                    (obj.predicted_position.x, obj.predicted_position.y),
                    radius=obj.prediction_range,
                    color=color,
                    thickness=3,
                )
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
                alpha = 0.5  # Transparency factor (0.0 - fully transparent, 1.0 - opaque)
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
            # --- Generate heatmap for this frame ---
            long_tracks = [obj for obj in tracker.current_potential_objects.union(tracker.all_objects) if hasattr(obj, 'history') and len([pt for pt in obj.history if pt is not None]) > 10]
            idx = len(frame_cache)
            heat = np.zeros((height, width), dtype=np.float32)
            arrow_overlay = np.zeros((height, width, 3), dtype=np.uint8)
            # --- Update direction grid for average heatmap ---
            for obj in long_tracks:
                for i in range(1, min(idx+1, len(obj.history))):
                    pt1 = obj.history[i-1]
                    pt2 = obj.history[i]
                    if pt1 is not None and pt2 is not None:
                        cx = (pt1.x + pt2.x) // 2
                        cy = (pt1.y + pt2.y) // 2
                        gx = min(cx // grid_size, grid_w - 1)
                        gy = min(cy // grid_size, grid_h - 1)
                        dx = pt2.x - pt1.x
                        dy = pt2.y - pt1.y
                        direction_sum_grid[gy, gx, 0] += dx
                        direction_sum_grid[gy, gx, 1] += dy
                        direction_count_grid[gy, gx] += 1
            for obj in long_tracks:
                if idx < len(obj.history):
                    pt = obj.history[idx]
                    if pt is not None:
                        cv2.circle(heat, (pt.x, pt.y), 8, (1,), -1)
                        # --- Update average heatmap incrementally ---
                        cv2.circle(avg_heat, (pt.x, pt.y), 8, (1,), -1)
                for i in range(1, min(idx+1, len(obj.history))):
                    pt1 = obj.history[i-1]
                    pt2 = obj.history[i]
                    if pt1 is not None and pt2 is not None:
                        cv2.arrowedLine(
                            arrow_overlay,
                            (pt1.x, pt1.y),
                            (pt2.x, pt2.y),
                            (0, 255, 255), 2, tipLength=0.4
                        )
            norm_heat = np.zeros_like(heat, dtype=np.uint8)
            cv2.normalize(heat, norm_heat, 0, 255, cv2.NORM_MINMAX)
            color_heat = cv2.applyColorMap(norm_heat, cv2.COLORMAP_JET)
            color_heat = cv2.addWeighted(color_heat, 1.0, arrow_overlay, 1.0, 0)
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
                    color_heat,  # store heatmap for this frame
                )
            )
            current_frame_idx += 1
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
                    color_heat,
                ) = frame_cache[current_frame_idx]
                display_frame_width = cached_frame.frame
            else:
                break

        # --- Draw play/pause button indicator ---
        status_text = "PAUSED" if not play else "PLAYING"
        overlay_text = (
            f"{status_text} | Space: Play/Pause | <-/->: Step | h: Toggle Heatmap | a: Toggle Avg Heatmap | Esc: Quit | Frame: {current_frame_idx}"
        )
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
        # --- Overlay heatmap if toggled ---
        if show_heatmap and 0 <= current_frame_idx < len(frame_cache):
            heat_overlay = frame_cache[current_frame_idx][7]
            overlay_frame = cv2.addWeighted(overlay_frame, 0.6, heat_overlay, 0.4, 0)
        # --- Overlay average heatmap if toggled ---
        if show_average_heatmap:
            arrow_overlay = overlay_frame.copy()
            fixed_length = 14  # Fixed arrow length for all cells
            for gy in range(grid_h):
                for gx in range(grid_w):
                    count = direction_count_grid[gy, gx]
                    if count > 0:
                        avg_dx = direction_sum_grid[gy, gx, 0] / count
                        avg_dy = direction_sum_grid[gy, gx, 1] / count
                        mag = np.hypot(avg_dx, avg_dy)
                        center_x = int((gx + 0.5) * grid_size)
                        center_y = int((gy + 0.5) * grid_size)
                        if mag > 1e-2:
                            dir_x = avg_dx / mag
                            dir_y = avg_dy / mag
                            tip_x = int(center_x + dir_x * fixed_length)
                            tip_y = int(center_y + dir_y * fixed_length)
                            cv2.arrowedLine(
                                arrow_overlay,
                                (center_x, center_y),
                                (tip_x, tip_y),
                                (0, 255, 255), 2, tipLength=0.3
                            )
                        else:
                            cv2.circle(arrow_overlay, (center_x, center_y), 2, (0, 255, 255), -1)
                    else:
                        center_x = int((gx + 0.5) * grid_size)
                        center_y = int((gy + 0.5) * grid_size)
                        cv2.circle(arrow_overlay, (center_x, center_y), 2, (0, 255, 255), -1)
            overlay_frame = cv2.addWeighted(overlay_frame, 0.7, arrow_overlay, 0.7, 0)
        display_frame = img_transformer.images_side_by_side(overlay_frame, objectsFrame, "Frame", "Objects")
        cv2.imshow("main", display_frame)

        display_frame_height, display_frame_width = display_frame.shape[:2]
        resize_window_to_screen("main", display_frame_width, display_frame_height)

        key = cv2.waitKey(0 if not play else 30)
        print(f"DEBUG: Key pressed: {key}")  # Add this line to debug key codes
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
                play = True
        elif key == ord('h') or key == ord('H') or key == 104:
            show_heatmap = not show_heatmap
        elif key == ord('a') or key == ord('A') or key == 97:
            show_average_heatmap = not show_average_heatmap
        if play and current_frame_idx < len(frame_cache) - 1:
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
