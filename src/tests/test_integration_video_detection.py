import os

import pytest

from batometer.objectfinder import ObjectFinder
from batometer.objectTracker import ObjectTracker
from batometer.videoManager import VideoManager


def count_detected_objects(video_path):
    """
    Helper function to count the number of detected objects in a video.

    Args:
        video_path (str): Path to the video file.

    Returns:
        int: Total number of detected objects across all frames.
    """
    with VideoManager(video_path) as video_manager:
        objectFinder = ObjectFinder()
        tracker = ObjectTracker(
            video_manager.width, video_manager.height
        )  # Pass width and height to ObjectTracker

        total_detections = 0

        while video_manager.has_more_frames():
            vid_frame = video_manager.read_frame()
            detections, _ = objectFinder.update(vid_frame)
            tracker.update(detections)
            total_detections += len(detections)

    return total_detections


@pytest.mark.parametrize(
    "video_path, obj_count",
    [
        ("/Users/tom/Code/Bat-O-Meter/videos/Snippets/Badger and Bug.mp4", 0),
    ],
)
def test_video_object_detection(video_path, obj_count):
    """
    Integration test to verify object detection in video files.

    Args:
        video_file (str): Path to the video file.
        obj_count (int): Number of objects expected to be detected.
    """
    total_detections = count_detected_objects(video_path)
    assert total_detections == obj_count, f"Expected at {obj_count} objects, but detected {total_detections}."
