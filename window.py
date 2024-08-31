import cv2
import numpy as np
import logging
import tkinter as tk

logger = logging.getLogger("Bat-O-Meter.window")


class ImageTransformer:
    TEXT_OFFSET = (1200, 50)
    TEXT_COLOUR = (0, 108, 255)
    TEXT_FONT = 0.8
    TEXT_FONT_FACE = cv2.FONT_HERSHEY_COMPLEX
    TEXT_THICKNESS = 2

    def __init__(self):
        self.window_width, self.window_height = self._compute_dimensions()

    def show_frame(self, window_name, img):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.window_width, self.window_height)
        cv2.imshow(window_name, img)
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            cv2.destroyWindow(window_name)
            logger.info("Quit the program as escape was pressed")
            quit()

    def overlay_two_images(self, background, overlay):
        background, overlay = background.copy(), overlay.copy()
        background, overlay = self._match_dimensions_to_img1(background, overlay)
        return self._overlay_transparent(background, overlay)

    def images_side_by_side(self, img1, img2, img1_text, img2_text):
        img1, img2 = self._match_dimensions_to_img1(img1, img2)
        cv2.putText(
            img1,
            img1_text,
            (img1.shape[1] - self.TEXT_OFFSET[0], img1.shape[0] - self.TEXT_OFFSET[1]),
            fontFace=self.TEXT_FONT_FACE,
            fontScale=self.TEXT_FONT,
            color=self.TEXT_COLOUR,
            thickness=self.TEXT_THICKNESS,
        )
        cv2.putText(
            img2,
            img2_text,
            (img2.shape[1] - self.TEXT_OFFSET[0], img2.shape[0] - self.TEXT_OFFSET[1]),
            fontFace=self.TEXT_FONT_FACE,
            fontScale=self.TEXT_FONT,
            color=self.TEXT_COLOUR,
            thickness=self.TEXT_THICKNESS,
        )
        concatenated = np.hstack((img1, img2))
        return concatenated

    def images_quadrant(
        self,
        top_left_img,
        top_right_img,
        bottom_left_img,
        bottom_right_img,
        top_left_img_text,
        top_right_img_text,
        bottom_left_img_text,
        bottom_right_img_text,
        center_text,
    ):
        top_left_img, top_right_img, bottom_left_img, bottom_right_img = (
            top_left_img.copy(),
            top_right_img.copy(),
            bottom_left_img.copy(),
            bottom_right_img.copy(),
        )
        top_row = self.images_side_by_side(top_left_img, top_right_img, top_left_img_text, top_right_img_text)
        bottom_row = self.images_side_by_side(
            bottom_left_img, bottom_right_img, bottom_left_img_text, bottom_right_img_text
        )
        final_image = np.vstack((top_row, bottom_row))
        cv2.putText(
            final_image,
            center_text,
            (int(final_image.shape[1] / 2), int(final_image.shape[0] / 2) + 20),
            fontFace=self.TEXT_FONT_FACE,
            fontScale=self.TEXT_FONT,
            color=self.TEXT_COLOUR,
            thickness=self.TEXT_THICKNESS,
        )
        return final_image

        # TODO: add mouse clicks to record position of birds
        # def mouse_callback(event, x, y, flags, params):
        #     global mouseX, mouseY
        #     if event == cv2.EVENT_LBUTTONDBLCLK:
        #         cv2.circle(final_image, (x, y), 100, (255, 0, 0), -1)
        #         mouseX, mouseY = x, y
        #     if event == 2:
        #         print(
        #             f"coords {x, y}, colors Blue- {final_image[y, x, 0]} , Green- {final_image[y, x, 1]}, Red- {final_image[y, x, 2]} "
        #         )

        # cv2.setMouseCallback(self.WINDOW_NAME, mouse_callback)

    def scale_frame_to_monitor(self, frame):
        height, width = frame.shape[:2]
        scale_factor = min(self.window_width / width, self.window_height / height)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        frame = cv2.resize(frame, (new_width, new_height))

    def _compute_dimensions(self):
        root = tk.Tk()
        root.withdraw()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()
        return screen_width - 100, screen_height - 100

    def _match_dimensions_to_img1(self, img1, img2):
        img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))
        if len(img1.shape) != len(img2.shape):
            logger.debug(f"Overlay shapes are not the same. Overlay {len(img1.shape)}, Background: {len(img2.shape)}")
        if len(img1.shape) == 2:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGRA2BGR)
        if len(img2.shape) == 2:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGRA2BGR)
        return img1, img2

    def _overlay_transparent(self, background, overlay):
        alpha = 0.5
        blended = cv2.addWeighted(overlay, alpha, background, 1 - alpha, 0)
        return blended
