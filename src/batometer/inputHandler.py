from batometer.constants import (
    ESC_KEYS,
    FLOW_OVERLAY_KEYS,
    HEATMAP_OVERLAY_KEYS,
    LEFT_KEYS,
    RIGHT_KEYS,
    SPACE_KEYS,
    TRACK_OVERLAY_KEYS,
)
from batometer.window import OverlayMode


class InputHandler:
    def __init__(self):
        self.is_autoplay = True
        self.overlay_mode = OverlayMode.NONE
        self.current_paused_frame_idx = 0
        self.show_objects = False

    def handle_key(self, key, frame_cache_length):
        match key:
            case k if k in ESC_KEYS:
                return "exit"
            case k if k in SPACE_KEYS:
                self.is_autoplay = not self.is_autoplay
            case k if k in LEFT_KEYS:
                if not self.is_autoplay and self.current_paused_frame_idx > 0:
                    self.current_paused_frame_idx -= 1
                elif self.is_autoplay:
                    self.is_autoplay = not self.is_autoplay
                    self.current_paused_frame_idx -= 1
            case k if k in RIGHT_KEYS:
                if not self.is_autoplay:
                    if self.current_paused_frame_idx < frame_cache_length - 1:
                        self.current_paused_frame_idx += 1
                    elif self.current_paused_frame_idx == frame_cache_length - 1:
                        self.is_autoplay = True
            case k if k in TRACK_OVERLAY_KEYS:
                self.overlay_mode = (
                    OverlayMode.TRACKS if self.overlay_mode != OverlayMode.TRACKS else OverlayMode.NONE
                )
            case k if k in FLOW_OVERLAY_KEYS:
                self.overlay_mode = (
                    OverlayMode.FLOW if self.overlay_mode != OverlayMode.FLOW else OverlayMode.NONE
                )
            case k if k in HEATMAP_OVERLAY_KEYS:
                self.overlay_mode = (
                    OverlayMode.HEATMAP if self.overlay_mode != OverlayMode.HEATMAP else OverlayMode.NONE
                )
            case k if k in [ord("o")]:
                self.show_objects = not self.show_objects
        return None
