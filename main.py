import cv2
import logging
from window import *
from utils import *
import os
import tempfile
from PIL import Image
import glob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Bat-O-Meter")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(dir=r"C:\Users\TSBus\OneDrive\Bat-O-Meter\batometer") as tmpdirname:
        logger.info(f"created temporary directory '{tmpdirname}'")

        VIDEO_PATH = r"C:\Users\TSBus\OneDrive\Bat-O-Meter\videos\batInThermal.mp4"
        video, width, height, fps = load_video(VIDEO_PATH)
        window = Window()

        backgroundSub = cv2.createBackgroundSubtractorMOG2(
            history=500,  # no. frames to keep
            varThreshold=100,  # asd
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
            frame_no = int(video.get(cv2.CAP_PROP_POS_FRAMES))
            ret, frame = video.read()
            og_frame = frame.copy()
            frame_time = caluclate_time(frame_no, fps)
            if not ret:
                break

            # Apply background subtractor to get the foreground maskv
            fgmask = backgroundSub.apply(frame)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            # Find contours in the mask
            contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # background_img = backgroundSub.getBackgroundImage()
            # cv2.imshow("Background", background_img)

            for contour in contours:
                # Get the centroid of the bird
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centroids.append((cx, cy))
                    logger.info(f"Found bird x: {cx} y: {cy} frame: {frame_no} time: {frame_time}")
                    cv2.drawContours(frame, contour, -1, (0, 255, 0), 3)
            final_image = window.display_images_side_by_side_with_overlay(
                og_frame, fgmask, frame, f"f: {frame_no} t: {frame_time}"
            )
            # cv2.imshow(window.WINDOW_NAME, final_image)
            # k = cv2.waitKey(0) & 0xFF
            # if k == 27:
            #     cv2.destroyWindow(window.WINDOW_NAME)
            #     logger.info("Quit the program as escape was pressed")
            #     quit()
            ext = "png"
            cv2.imwrite(str(os.path.join(tmpdirname, f"frame-{frame_no}.{ext}")), final_image)
            if frame_no > 200:
                break

        def create_gif_from_images(save_path: str, image_path: str, ext: str) -> None:
            ext = ext.replace(".", "")
            image_paths = sorted(glob.glob(os.path.join(image_path, f"*.{ext}")))
            image_paths.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
            pil_images = [Image.open(im_path) for im_path in image_paths]

            pil_images[0].save(
                save_path,
                format="GIF",
                append_images=pil_images,
                save_all=True,
                duration=50,
                loop=0,
            )

        create_gif_from_images(
            r"C:\Users\TSBus\OneDrive\Bat-O-Meter\batometer\output.gif",
            tmpdirname,
            ext,
        )
