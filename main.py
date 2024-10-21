import logging
from typing import Tuple

import cv2
from cv2.typing import MatLike

from constants import *
from frame import Frame
from objectfinder import Object, ObjectFinder
from tracker import *
from utils import *
from window import *

FORMAT = "%(asctime)s %(levelname)-8s %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(BATOMETER)


if __name__ == "__main__":
    VIDEO_PATH = r"C:\Users\TSBus\OneDrive\Bat-O-Meter\videos\batInThermal.mp4"
    video, width, height, fps, noFrames = load_video(VIDEO_PATH)
    objectFinder = ObjectFinder()
    # objectFinder.initialise(video)
    tracker = EuclideanDistTracker()
    img_transformer = ImageTransformer()

    while video.isOpened():
        ret, vid_frame = video.read()
        frame_num = int(video.get(cv2.CAP_PROP_POS_FRAMES))
        if not ret:
            logger.error(f"Can't receive frame (stream end?). FrameNum: {frame_num}. Exiting ...")
            break
        frame_time = caluclate_video_time_from_frame_num(frame_num, fps)
        frame = Frame(vid_frame, frame_num)
        logger.info(f"Opened frame: {frame_num}/{noFrames} - time: {frame_time}")

        # Find objects
        detections = objectFinder.update(frame)

        # Clean bogus detections by looking in the future
        next_frame = frame_num + 1
        video.set(cv2.CAP_PROP_POS_FRAMES, next_frame)
        _, fut_frame = video.read()
        future_detections = objectFinder.update(Frame(fut_frame, next_frame))
        cleaned_detections = []
        for detection in detections:
            x1, y1, _, _ = detection
            for fut_detection in future_detections:
                x2, y2, _, _ = fut_detection
                dist = math.hypot(x1 - x2, y1 - y2)
                if dist < 100:
                    cleaned_detections.append(detection)
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

        # Track objects
        tracked_detections = tracker.update(cleaned_detections)

        # Display tracked detections
        for detection in tracked_detections:
            x, y, w, h, id = detection
            cv2.putText(
                frame.frame,
                str(id),
                (x, y - 15),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (255, 0, 0),
                2,
            )
            cv2.rectangle(frame.frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.imshow("main", frame.frame)
        key = cv2.waitKey(30)
        if key == 27:
            break
        # contour_overlay = img_transformer.overlay_two_images(og_frame, contour_frame)
        # final_image = img_transformer.images_quadrant(
        #     og_frame,
        #     fgmask,
        #     contour_frame,
        #     contour_overlay,
        #     "Original Video",
        #     "Foreground Mask",
        #     "Original Video + Foreground Mask",
        #     "Original Video + Contours",
        #     f"f: {frame_num} t: {frame_time}",
        # )
        # img_transformer.show_frame("Main", final_image)

        # save_image_to_temp(final_image, frame_num)

# images_to_mp4(
#     r"C:\Users\TSBus\OneDrive\Bat-O-Meter\batometer\.temp",
#     r"C:\Users\TSBus\OneDrive\Bat-O-Meter\batometer\video.mp4",
#     30,
# )
