import cv2
import numpy as np
import logging
import tkinter as tk

logger = logging.getLogger("Bat-O-Meter.window")


class Window:
    WINDOW_NAME = "Image"

    def __init__(self):
        self.window_width, self.window_height = self._compute_dimensions()

    def display_images_as_overlay(self, background, overlay):
        background, overlay = self._match_dimensions(background, overlay)
        cv2.imshow(self.WINDOW_NAME, self._overlay_transparent(background, overlay))
        cv2.waitKey(0)
        cv2.destroyWindow(self.WINDOW_NAME)

    def display_images_side_by_side(self, background, overlay):
        background, overlay = self._match_dimensions(background, overlay)
        concatenated = np.hstack((background, overlay))
        cv2.imshow(self.WINDOW_NAME, concatenated)
        cv2.waitKey(0)
        cv2.destroyWindow(self.WINDOW_NAME)

    def display_images_side_by_side_with_overlay(
        self, original, backgroundSubtractor, contour, extra_message
    ):
        TEXT_OFFSET = (1200, 50)
        TEXT_COLOUR = (0, 108, 255)
        TEXT_FONT = 0.8
        TEXT_FONT_FACE = cv2.FONT_HERSHEY_COMPLEX
        TEXT_THICKNESS = 2
        original, backgroundSubtractor, contour = (
            original.copy(),
            backgroundSubtractor.copy(),
            contour.copy(),
        )
        original, backgroundSubtractor = self._match_dimensions(
            original, backgroundSubtractor
        )
        backgroundSubtractor = cv2.cvtColor(backgroundSubtractor, cv2.COLOR_BGR2HSV)
        cv2.putText(
            contour,
            "Contour",
            (contour.shape[1] - TEXT_OFFSET[0], contour.shape[0] - TEXT_OFFSET[1]),
            fontFace=TEXT_FONT_FACE,
            fontScale=TEXT_FONT,
            color=TEXT_COLOUR,
            thickness=TEXT_THICKNESS,
        )
        overlay = self._overlay_transparent(original, backgroundSubtractor)
        overlay = self._overlay_transparent(overlay, contour)

        cv2.putText(
            original,
            "Original",
            (original.shape[1] - TEXT_OFFSET[0], original.shape[0] - TEXT_OFFSET[1]),
            fontFace=TEXT_FONT_FACE,
            fontScale=TEXT_FONT,
            color=TEXT_COLOUR,
            thickness=TEXT_THICKNESS,
        )
        cv2.putText(
            backgroundSubtractor,
            "BackgroundSubtractor",
            (
                backgroundSubtractor.shape[1] - TEXT_OFFSET[0],
                backgroundSubtractor.shape[0] - TEXT_OFFSET[1],
            ),
            fontFace=TEXT_FONT_FACE,
            fontScale=TEXT_FONT,
            color=TEXT_COLOUR,
            thickness=TEXT_THICKNESS,
        )
        cv2.putText(
            overlay,
            "Overlay",
            (overlay.shape[1] - TEXT_OFFSET[0], overlay.shape[0] - TEXT_OFFSET[1]),
            fontFace=TEXT_FONT_FACE,
            fontScale=TEXT_FONT,
            color=TEXT_COLOUR,
            thickness=TEXT_THICKNESS,
        )
        top_row = np.hstack((original, backgroundSubtractor))
        bottom_row = np.hstack((overlay, contour))
        final_image = np.vstack((top_row, bottom_row))
        height, width = final_image.shape[:2]
        scale_factor = min(self.window_width / width, self.window_height / height)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        final_image = cv2.resize(final_image, (new_width, new_height))
        cv2.putText(
            final_image,
            extra_message,
            (int(final_image.shape[1] / 2), int(final_image.shape[0] / 2) + 20),
            fontFace=TEXT_FONT_FACE,
            fontScale=TEXT_FONT,
            color=TEXT_COLOUR,
            thickness=TEXT_THICKNESS,
        )
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME, self.window_width, self.window_height)

        # TODO: add mouse clicks to record position of birds
        def mouse_callback(event, x, y, flags, params):
            global mouseX, mouseY
            if event == cv2.EVENT_LBUTTONDBLCLK:
                cv2.circle(final_image, (x, y), 100, (255, 0, 0), -1)
                mouseX, mouseY = x, y
            if event == 2:
                print(
                    f"coords {x, y}, colors Blue- {final_image[y, x, 0]} , Green- {final_image[y, x, 1]}, Red- {final_image[y, x, 2]} "
                )

        cv2.setMouseCallback(self.WINDOW_NAME, mouse_callback)
        cv2.imshow(self.WINDOW_NAME, final_image)
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            cv2.destroyWindow(self.WINDOW_NAME)
            logger.info("Quit the program as escape was pressed")
            quit()

    def _compute_dimensions(self):
        root = tk.Tk()
        root.withdraw()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()
        return screen_width - 100, screen_height - 100

    def _match_dimensions(self, img1, img2):
        img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))
        if len(img1.shape) != len(img2.shape):
            logger.debug(
                f"Overlay shapes are not the same. Overlay {len(img1.shape)}, Background: {len(img2.shape)}"
            )
        if len(img1.shape) == 2:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGRA2BGR)
        if len(img2.shape) == 2:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGRA2BGR)
        return img1, img2

    def _overlay_transparent(self, background, overlay):
        background, overlay = background.copy(), overlay.copy()
        alpha = 0.5
        blended = cv2.addWeighted(overlay, alpha, background, 1 - alpha, 0)
        return blended
