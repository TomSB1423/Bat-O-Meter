import cv2
import numpy as np

from batometer.detectionObject import IdentifiedObject

ARROW_LENGTH = 14  # pixels
ARROW_COLOR = (0, 255, 255)
ARROW_THICKNESS = 2
ARROW_TIP_LENGTH = 0.3
DOT_RADIUS = 2
DOT_COLOR = (0, 255, 255)
HEATMAP_ALPHA = 0.4
FRAME_ALPHA = 0.6
GRID_OVERLAY_ALPHA = 0.7
DIRECTION_THRESHOLD = 1e-2


class Heatmap:
    def __init__(self, width: int, height: int):
        self.grid_size = 32  # You can adjust this for finer/coarser arrows
        self.grid_h = height // self.grid_size
        self.grid_w = width // self.grid_size
        self.direction_sum_grid = np.zeros(
            (self.grid_h, self.grid_w, 2), dtype=np.float32
        )  # (sum_dx, sum_dy) per cell
        self.direction_count_grid = np.zeros((self.grid_h, self.grid_w), dtype=np.int32)  # count per cell

    def update(self, tracks: set[IdentifiedObject]):
        for obj in tracks:
            for i in range(1, len(obj.history)):
                point1 = obj.history[i - 1]
                point2 = obj.history[i]
                if point1 is not None and point2 is not None:
                    mid_x = (point1.x + point2.x) // 2
                    mid_y = (point1.y + point2.y) // 2
                    grid_x = min(mid_x // self.grid_size, self.grid_w - 1)
                    grid_y = min(mid_y // self.grid_size, self.grid_h - 1)
                    dx = point2.x - point1.x
                    dy = point2.y - point1.y
                    self.direction_sum_grid[grid_y, grid_x, 0] += dx
                    self.direction_sum_grid[grid_y, grid_x, 1] += dy
                    self.direction_count_grid[grid_y, grid_x] += 1

    def create_flow_overlay(self, frame: "cv2.typing.MatLike") -> "cv2.typing.MatLike":
        arrow_overlay = frame.copy()
        for gy in range(self.grid_h):
            for gx in range(self.grid_w):
                count = self.direction_count_grid[gy, gx]
                center_x = int((gx + 0.5) * self.grid_size)
                center_y = int((gy + 0.5) * self.grid_size)
                avg_dx = self.direction_sum_grid[gy, gx, 0] / count
                avg_dy = self.direction_sum_grid[gy, gx, 1] / count
                mag = np.hypot(avg_dx, avg_dy)
                if mag > DIRECTION_THRESHOLD:
                    dir_x = avg_dx / mag
                    dir_y = avg_dy / mag
                    tip_x = int(center_x + dir_x * ARROW_LENGTH)
                    tip_y = int(center_y + dir_y * ARROW_LENGTH)
                    cv2.arrowedLine(
                        arrow_overlay,
                        (center_x, center_y),
                        (tip_x, tip_y),
                        ARROW_COLOR,
                        ARROW_THICKNESS,
                        tipLength=ARROW_TIP_LENGTH,
                    )
                else:
                    cv2.circle(arrow_overlay, (center_x, center_y), DOT_RADIUS, DOT_COLOR, -1)
        return arrow_overlay
