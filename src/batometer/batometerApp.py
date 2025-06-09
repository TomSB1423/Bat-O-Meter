import os

import cv2
import pandas as pd
import logging 

from .frameCache import FrameCacheEntry
from .heatmap import Heatmap
from .inputHandler import InputHandler
from .objectfinder import ObjectFinder
from .objectTracker import ObjectTracker
from .videoManager import VideoManager
from .constants import BATOMETER
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


class BatometerApp:
    def __init__(self, video_path):
        self.video_path = video_path
        self.objectFinder = ObjectFinder()
        self.img_transformer = ImageTransformer()
        self.input_handler = InputHandler()
        self.frame_cache: list[FrameCacheEntry] = []
        self.window_name = "Batometer"

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        with VideoManager(self.video_path) as video_manager:
            heatmap = Heatmap(video_manager.width, video_manager.height)
            tracker = ObjectTracker(video_manager.width, video_manager.height)
            logger.info(f"Width: {video_manager.width} Height: {video_manager.height}")

            while video_manager.has_more_frames():
                if self.input_handler.is_autoplay:
                    frame = video_manager.read_frame()
                    original_frame = frame.copy()

                    # Identify objects
                    detections, objects_frame = self.objectFinder.update(frame)
                    tracked_detections, predicted_objs = tracker.update(detections)
                    heatmap.update(tracked_detections)

                    # Draw tracks on frame
                    for detection in detections:
                        draw_detection_rectangle(frame, detection)
                    for obj in predicted_objs:
                        draw_predicted_object(frame, obj)
                    for obj in tracked_detections:
                        draw_tracked_object(frame, obj)

                    # Create overlays
                    tracker_overlay = tracker.create_overlay(frame)
                    heatmap_overlay = tracker.create_heatmap_overlay(original_frame)
                    flow_overlay = heatmap.create_flow_overlay(frame)

                    # Add to cache
                    self.frame_cache.append(
                        FrameCacheEntry(
                            frame.copy(),
                            objects_frame.copy(),
                            tracker_overlay.copy(),
                            flow_overlay.copy(),
                            heatmap_overlay.copy(),
                            video_manager.frame_num,
                            video_manager.frame_time,
                            detections,
                            tracked_detections,
                        )
                    )
                    self.input_handler.current_paused_frame_idx = video_manager.frame_num - 1
                    
                    
                    # Generate YOLO training data
                    txt_data = []
                    for obj in tracked_detections:
                        if len(obj.history) > 10:
                            txt_data.append(f"0 {obj.point.x / video_manager.width} {obj.point.y / video_manager.width} {obj.width / video_manager.width} {obj.height / video_manager.width}")

                    val_output_folder = "/Users/tom/Code/Bat-O-Meter/src/yolo/labels"
                    png_output_folder = "/Users/tom/Code/Bat-O-Meter/src/yolo/images"
                    if video_manager.frame_num < 554:
                        val_output_folder += "/val"
                        png_output_folder += "/val"
                        os.makedirs(val_output_folder, exist_ok=True)
                        os.makedirs(png_output_folder, exist_ok=True)
                    else:
                        val_output_folder += "/train"
                        png_output_folder += "/train"
                        os.makedirs(val_output_folder, exist_ok=True)
                        os.makedirs(png_output_folder, exist_ok=True)
                        
                    png_output_path = os.path.join(png_output_folder, f"frame_{video_manager.frame_num}.png")
                    val_output_folder = os.path.join(val_output_folder, f"frame_{video_manager.frame_num}.txt")

                    with open(val_output_folder, "w") as txt_file:
                        txt_file.write("\n".join(txt_data))
                    cv2.imwrite(png_output_path, original_frame)

                else:
                    frame_cache_entry = self.frame_cache[self.input_handler.current_paused_frame_idx]
                    frame = frame_cache_entry.video_frame
                    objects_frame = frame_cache_entry.objects_frame  # Ensure objects_frame is defined

                frame = draw_overlay_text(
                    frame, self.input_handler.is_autoplay, self.input_handler.current_paused_frame_idx, video_manager.max_frames
                )
                match self.input_handler.overlay_mode:
                    case OverlayMode.TRACKS:
                        video_overlay_frame = self.frame_cache[
                            self.input_handler.current_paused_frame_idx
                        ].tracks_frame
                    case OverlayMode.FLOW:
                        video_overlay_frame = self.frame_cache[
                            self.input_handler.current_paused_frame_idx
                        ].flow_frame
                    case OverlayMode.HEATMAP:
                        video_overlay_frame = self.frame_cache[
                            self.input_handler.current_paused_frame_idx
                        ].heatmap_frame
                    case _:
                        video_overlay_frame = frame

                if not self.input_handler.show_objects:
                    display_frame = video_overlay_frame
                else:
                    display_frame = self.img_transformer.images_side_by_side(
                        video_overlay_frame, objects_frame, "Frame", "Objects"
                    )
                cv2.imshow(self.window_name, display_frame)

                # Only set window size once, after window creation and first frame
                if self.input_handler.current_paused_frame_idx == 0:
                    display_frame_height, display_frame_width = display_frame.shape[:2]
                    resize_window_to_screen(self.window_name, display_frame_width, display_frame_height)

                key = cv2.waitKey(0 if not self.input_handler.is_autoplay else 30)
                action = self.input_handler.handle_key(key, len(self.frame_cache))
                if action == "exit":
                    break

        # Define a helper function to map directions
        def get_direction(start, end):
            dx = end[0] - start[0]
            dy = end[1] - start[1]

            if abs(dx) > abs(dy):
                if dx > 0:
                    return "right"
                else:
                    return "left"
            else:
                if dy > 0:
                    return "down"
                else:
                    return "up"

        excel_data = []
        for obj in tracker.all_objects:
            if len(obj.history) > 10:
                # Filter out None values from history
                valid_history = [point for point in obj.history if point is not None]

                if len(valid_history) > 10:
                    # Calculate average position of the first 20 frames
                    first_20 = valid_history[:20]
                    avg_incoming_x = sum(point.x for point in first_20) / len(first_20)
                    avg_incoming_y = sum(point.y for point in first_20) / len(first_20)

                    # Calculate average position of the last 20 frames
                    last_20 = valid_history[-20:]
                    avg_outgoing_x = sum(point.x for point in last_20) / len(last_20)
                    avg_outgoing_y = sum(point.y for point in last_20) / len(last_20)

                    # Translate directions to descriptive terms
                    incoming_desc = get_direction(
                        (avg_incoming_x, avg_incoming_y), (avg_outgoing_x, avg_outgoing_y)
                    )
                    outgoing_desc = get_direction(
                        (avg_outgoing_x, avg_outgoing_y), (avg_incoming_x, avg_incoming_y)
                    )

                    # Calculate likelihood of being a bat based on movement
                    movement_pattern = [
                        (
                            valid_history[i + 1].x - valid_history[i].x,
                            valid_history[i + 1].y - valid_history[i].y,
                        )
                        for i in range(len(valid_history) - 1)
                    ]
                    likelihood_bat = tracker.calculate_bat_likelihood(movement_pattern)

                    # Store data for Excel
                    obj_data = {
                        "Object ID": obj.id,
                        "Incoming Direction": incoming_desc,
                        "Outgoing Direction": outgoing_desc,
                        "Likelihood of Bat": likelihood_bat,
                    }
                    excel_data.append(obj_data)
        if excel_data:
            df = pd.DataFrame(excel_data)
            output_path = "bat_analysis.csv"
            df.to_csv(output_path, index=False)
            print(f"Excel spreadsheet saved to {output_path}")

        # Save
        heatmap_output_path = "heatmap.png"
        cv2.imwrite(
            heatmap_output_path, self.frame_cache[self.input_handler.current_paused_frame_idx].heatmap_frame
        )

        cv2.destroyAllWindows()
