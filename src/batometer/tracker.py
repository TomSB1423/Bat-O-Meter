import logging
import math
from typing import List

from .constants import BATOMETER
from .detectionObject import Detection, IdentifiedObject, Point

logger = logging.getLogger(f"{BATOMETER}.EuclideanDistTracker")


class EuclideanDistTracker:
    """
    Tracks objects across video frames using Euclidean distance between their center points.
    Assigns unique IDs to detected objects and maintains their identities across frames.
    """

    def __init__(self) -> None:
        """
        Initializes the tracker with an empty dictionary for center points and an ID counter.
        """
        # Store the center positions of the objects
        self.all_tracks: List[IdentifiedObject] = []
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count: int = 0
        self.last_seen: dict[int, int] = {}  # id -> last seen frame number

    def update(self, detected_objects: List[Detection], frame_num: int) -> List[IdentifiedObject]:
        """
        Updates the tracker with new detections, assigns IDs, and returns identified objects.

        Args:
            detected_objects (List[Detection]): List of detected objects in the current frame.
            frame_num (int): The current frame number (for tracking missed frames).

        Returns:
            List[IdentifiedObject]: List of objects with assigned unique IDs.
        """
        DIST_THRESHOLD = 50
        DIST_THRESHOLD_MISSED = 100  # Increased threshold for missed objects
        matched_tracks = set()
        currently_tracked: List[IdentifiedObject] = []

        for det in detected_objects:
            min_dist = float('inf')
            matched_track = None
            for track in self.all_tracks:
                if track in matched_tracks:
                    continue
                last_seen = self.last_seen.get(track.id, frame_num)
                missed_frames = frame_num - last_seen
                threshold = DIST_THRESHOLD if missed_frames <= 1 else DIST_THRESHOLD_MISSED
                dist = math.hypot(det.x - track.x, det.y - track.y)
                if dist < threshold and dist < min_dist:
                    min_dist = dist
                    matched_track = track
            if matched_track is not None:
                matched_track.update_location(Point(det.x, det.y))
                currently_tracked.append(matched_track)
                matched_tracks.add(matched_track)
                self.last_seen[matched_track.id] = frame_num
            else:
                new_obj = IdentifiedObject(self.id_count, det)
                self.id_count += 1
                self.all_tracks.append(new_obj)
                currently_tracked.append(new_obj)
                matched_tracks.add(new_obj)
                self.last_seen[new_obj.id] = frame_num

        # Optionally, you could remove tracks not seen for many frames here
        MAX_MISSED_FRAMES = 10
        # Remove tracks that have not been seen for more than MAX_MISSED_FRAMES
        self.all_tracks = [
            track for track in self.all_tracks
            if (frame_num - self.last_seen.get(track.id, frame_num)) <= MAX_MISSED_FRAMES
        ]
        # Also clean up last_seen dict to only keep active tracks
        active_ids = {track.id for track in self.all_tracks}
        self.last_seen = {tid: fnum for tid, fnum in self.last_seen.items() if tid in active_ids}
        return currently_tracked
