import os
import pytest
from batometer.main import main
from batometer.utils import load_video
from batometer.objectfinder import ObjectFinder
from batometer.tracker import EuclideanDistTracker

def count_detected_objects(video_path):
    """
    Helper function to count the number of detected objects in a video.

    Args:
        video_path (str): Path to the video file.

    Returns:
        int: Total number of detected objects across all frames.
    """
    video, width, height, fps, noFrames = load_video(video_path)
    objectFinder = ObjectFinder()
    tracker = EuclideanDistTracker()

    total_detections = 0

    while True:
        ret, vid_frame = video.read()
        if not ret:
            break

        detections, _ = objectFinder.update(vid_frame)
        tracker.update(detections)
        total_detections += len(detections)

    video.release()
    return total_detections

@pytest.mark.parametrize("video_file, expected_min_objects", [
    ("videos/Snippets/Bat - foraging Dark Background .mp4", 5),
    ("videos/Snippets/BirdofPrey - Flying.mp4", 3),
    ("videos/Snippets/Rabbit.mp4", 2),
])
def test_video_object_detection(video_file, expected_min_objects):
    """
    Integration test to verify object detection in video files.

    Args:
        video_file (str): Path to the video file.
        expected_min_objects (int): Minimum number of objects expected to be detected.
    """
    video_path = os.path.join(os.getcwd(), video_file)
    assert os.path.exists(video_path), f"Video file {video_path} does not exist."

    total_detections = count_detected_objects(video_path)
    assert total_detections >= expected_min_objects, (
        f"Expected at least {expected_min_objects} objects, but detected {total_detections}."
    )
