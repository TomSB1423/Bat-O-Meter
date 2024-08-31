import cv2
import logging
from window import *
from utils import *

FORMAT = "%(asctime)s %(levelname)-8s %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger("Bat-O-Meter")


if __name__ == "__main__":
    VIDEO_PATH = r"C:\Users\TSBus\OneDrive\Bat-O-Meter\videos\batInThermal.mp4"
    video, width, height, fps, noFrames = load_video(VIDEO_PATH)
    img_transformer = ImageTransformer()

    backgroundSub = cv2.createBackgroundSubtractorMOG2(
        history=500,  # no. frames to keep
        varThreshold=100,  # sensitivity of
        detectShadows=False,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Priming with initial frames (background should be stable)
    logger.info("Priming background subtractor...")
    initial_frame_count = 500
    for _ in range(initial_frame_count):
        ret, frame = video.read()
        if not ret:
            break
        # Update the background model with initial frames
        backgroundSub.apply(frame)
    logger.info("Successfully primed background subtractor")

    video.set(cv2.CAP_PROP_POS_FRAMES, 1)
    centroids = []
    while video.isOpened():
        frame_num = int(video.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = video.read()
        if not ret:
            break
        og_frame = frame.copy()
        frame_time = caluclate_video_time_from_frame_num(frame_num, fps)
        logger.info(f"Opened frame: {frame_num}/{noFrames} - time: {frame_time}")
        

        # Apply background subtractor to get the foreground maskv
        fgmask = backgroundSub.apply(frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        # Find contours in the mask
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # background_img = backgroundSub.getBackgroundImage()
        # cv2.imshow("Background", background_img)

        contour_frame = frame.copy()
        contours_poly = [None] * len(contours)
        boundRect = [None] * len(contours)
        for i, contour in enumerate(contours):
            # Get the centroid of the bird
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))
                logger.debug(f"Found bird x: {cx} y: {cy} frame: {frame_num} time: {frame_time}")
                # cv2.drawContours(contour_frame, contour, -1, (0, 255, 0), 3)
                contours_poly[i] = cv2.approxPolyDP(contour, 3, True)
                boundRect[i] = cv2.boundingRect(contours_poly[i])
                cv2.rectangle(
                    contour_frame,
                    (int(boundRect[i][0]), int(boundRect[i][1])),
                    (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])),
                    (72, 249, 125),
                    2,
                )
        contour_overlay = img_transformer.overlay_two_images(og_frame, contour_frame)
        final_image = img_transformer.images_quadrant(
            og_frame,
            fgmask,
            contour_frame,
            contour_overlay,
            "Original Video",
            "Foreground Mask",
            "Original Video + Foreground Mask",
            "Original Video + Contours",
            f"f: {frame_num} t: {frame_time}",
        )
        save_image_to_temp(final_image, frame_num)
        # img_transformer.show_frame("Main", final_image)

images_to_mp4(
    r"C:\Users\TSBus\OneDrive\Bat-O-Meter\batometer\.temp",
    r"C:\Users\TSBus\OneDrive\Bat-O-Meter\batometer\video.mp4",
    30,
)
