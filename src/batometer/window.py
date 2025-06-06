import logging
import tkinter as tk
from enum import Enum

import cv2
import numpy as np
from cv2.typing import MatLike


logger = logging.getLogger("Bat-O-Meter.window")


class OverlayMode(Enum):
    NONE = "none"
    TRACKS = "heatmap"
    HEATMAP = "average"


class ImageTransformer:
    """
    Provides image transformation and display utilities for Bat-O-Meter visualizations.
    """

    TEXT_OFFSET: tuple[int, int] = (1200, 50)
    TEXT_COLOUR: tuple[int, int, int] = (0, 108, 255)
    TEXT_FONT: float = 0.8
    TEXT_FONT_FACE: int = cv2.FONT_HERSHEY_COMPLEX
    TEXT_THICKNESS: int = 2

    def __init__(self) -> None:
        """
        Initializes the ImageTransformer and computes window dimensions.
        """
        self.window_width, self.window_height = self._compute_dimensions()

    def show_frame(self, window_name: str, img: "MatLike") -> None:
        """
        Displays a frame in a resizable OpenCV window.

        Args:
            window_name (str): Name of the window.
            img (MatLike): Image to display.
        """
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.window_width, self.window_height)
        cv2.imshow(window_name, img)
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            cv2.destroyWindow(window_name)
            logger.info("Quit the program as escape was pressed")
            quit()

    def overlay_two_images(self, background: "MatLike", overlay: "MatLike") -> "MatLike":
        """
        Overlays one image on top of another, matching dimensions as needed.

        Args:
            background (MatLike): Background image.
            overlay (MatLike): Overlay image.

        Returns:
            MatLike: Combined image.
        """
        background, overlay = background.copy(), overlay.copy()
        background, overlay = self._match_dimensions_to_img1(background, overlay)
        return self._overlay_transparent(background, overlay)

    def images_side_by_side(
        self, img1: "MatLike", img2: "MatLike", img1_text: str, img2_text: str
    ) -> "MatLike":
        """
        Places two images side by side with optional text labels.

        Args:
            img1 (MatLike): First image.
            img2 (MatLike): Second image.
            img1_text (str): Label for the first image.
            img2_text (str): Label for the second image.

        Returns:
            MatLike: Combined image with both images side by side.
        """
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
        top_left_img: MatLike,
        top_right_img: MatLike,
        bottom_left_img: MatLike,
        bottom_right_img: MatLike,
        top_left_img_text: str,
        top_right_img_text: str,
        bottom_left_img_text: str,
        bottom_right_img_text: str,
        center_text: str,
    ) -> MatLike:
        """
        Arranges four images in a quadrant layout with text labels and a center label.

        Args:
            top_left_img (MatLike): Top-left image.
            top_right_img (MatLike): Top-right image.
            bottom_left_img (MatLike): Bottom-left image.
            bottom_right_img (MatLike): Bottom-right image.
            top_left_img_text (str): Label for top-left image.
            top_right_img_text (str): Label for top-right image.
            bottom_left_img_text (str): Label for bottom-left image.
            bottom_right_img_text (str): Label for bottom-right image.
            center_text (str): Center label text.

        Returns:
            MatLike: Quadrant-arranged image.
        """
        top_left_img, top_right_img, bottom_left_img, bottom_right_img = (
            top_left_img.copy(),
            top_right_img.copy(),
            bottom_left_img.copy(),
            bottom_right_img.copy(),
        )
        top_row = self.images_side_by_side(top_left_img, top_right_img, top_left_img_text, top_right_img_text)
        bottom_row = self.images_side_by_side(
            bottom_left_img,
            bottom_right_img,
            bottom_left_img_text,
            bottom_right_img_text,
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

    def scale_frame_to_monitor(self, frame: MatLike) -> None:
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

    def _match_dimensions_to_img1(self, img1: MatLike, img2: MatLike):
        img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))
        if len(img1.shape) != len(img2.shape):
            logger.debug(
                f"Overlay shapes are not the same. Overlay {len(img1.shape)}, "
                f"Background: {len(img2.shape)}"
            )
        if len(img1.shape) == 2:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGRA2BGR)
        if len(img2.shape) == 2:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGRA2BGR)
        return img1, img2

    def _overlay_transparent(self, background: MatLike, overlay: MatLike) -> MatLike:
        alpha = 0.5
        blended = cv2.addWeighted(overlay, alpha, background, 1 - alpha, 0)
        return blended


def resize_window_to_screen(window_name: str, width: int, height: int):
    """
    Resize the OpenCV window based on screen size and video aspect ratio.
    """
    # Use tkinter to get screen size (cross-platform, no subprocess)
    root = tk.Tk()
    root.withdraw()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
    scale = min(screen_width / width, screen_height / height, 0.8)
    window_w = int(width * scale)
    window_h = int(height * scale)
    cv2.resizeWindow(window_name, window_w, window_h)


def draw_overlay_text(
    overlay_frame: "cv2.typing.MatLike", is_play: bool, current_frame_idx: int
) -> "cv2.typing.MatLike":
    """
    Draws overlay text (status, controls, frame number) at the top of the frame.

    Args:
        overlay_frame ("cv2.typing.MatLike"): The frame on which to draw the overlay text.
        is_play (bool): Is the video in play mode.
        current_frame_idx (int): The index of the current frame (for display purposes).

    Returns:
        "cv2.typing.MatLike": The frame with the overlay text drawn on it.
    """
    status_text = "PAUSED" if not is_play else "PLAYING"
    overlay_text = f"{status_text} | Space: Play/Pause | <-/->: Step | t: Toggle Tracks | a: Toggle Avg Heatmap | Esc: Quit | Frame: {current_frame_idx}"
    cv2.rectangle(overlay_frame, (0, 0), (overlay_frame.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(
        overlay_frame,
        overlay_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return overlay_frame


def draw_detection_rectangle(frame: "cv2.typing.MatLike", detection) -> None:
    """
    Draws a red rectangle for a detection on the frame.
    """
    cv2.rectangle(
        frame,
        (detection.point.x, detection.point.y),
        (detection.point.x + detection.width, detection.point.y + detection.height),
        (0, 0, 255),
        3,
    )


def draw_tracked_object(frame: "cv2.typing.MatLike", obj) -> None:
    """
    Draws tracked object prediction, history, and bounding box on the frame.
    """
    color = ((obj.id * 70) % 256, (obj.id * 150) % 256, (obj.id * 230) % 256)
    overlay = frame.copy()
    cv2.circle(
        overlay,
        (obj.predicted_position.x, obj.predicted_position.y),
        radius=obj.prediction_range,
        color=color,
        thickness=-1,
    )
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    if hasattr(obj, "history") and len(obj.history) > 1:
        last_valid_point = None
        for i in range(1, len(obj.history)):
            pt1 = obj.history[i - 1]
            pt2 = obj.history[i]
            if pt1 is not None and pt2 is not None:
                cv2.line(frame, (pt1.x, pt1.y), (pt2.x, pt2.y), color, 2)
                last_valid_point = pt2
            elif pt2 is not None and last_valid_point is not None:
                cv2.line(
                    frame,
                    (last_valid_point.x, last_valid_point.y),
                    (pt2.x, pt2.y),
                    color,
                    2,
                )
                last_valid_point = pt2
            elif pt2 is not None:
                last_valid_point = pt2
    cv2.putText(
        frame,
        str(obj.id),
        (obj.point.x, obj.point.y - 15),
        cv2.FONT_HERSHEY_PLAIN,
        2,
        color,
        2,
    )
    cv2.rectangle(
        frame,
        (obj.point.x, obj.point.y),
        (obj.point.x + obj.width, obj.point.y + obj.height),
        color,
        3,
    )


def draw_predicted_object(frame: "cv2.typing.MatLike", obj) -> None:
    """
    Draws a circle for a predicted object on the frame.
    """
    color = ((obj.id * 70) % 256, (obj.id * 150) % 256, (obj.id * 230) % 256)
    cv2.circle(
        frame,
        (obj.predicted_position.x, obj.predicted_position.y),
        radius=obj.prediction_range,
        color=color,
        thickness=3,
    )
