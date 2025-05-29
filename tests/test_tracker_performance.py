import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from detectionObject import DetectionObject
from tracker import EuclideanDistTracker
from visualization import visualize_tracking


def test_tracker_tracks_moving_object(request):
    """
    Test that the tracker assigns a consistent ID to a moving object across frames.
    """
    tracker = EuclideanDistTracker()
    # Simulate a moving object across 5 frames
    detections_per_frame = [
        [DetectionObject(x=10, y=10, width=5, height=5)],
        [DetectionObject(x=15, y=12, width=5, height=5)],
        [DetectionObject(x=20, y=15, width=5, height=5)],
        [DetectionObject(x=25, y=18, width=5, height=5)],
        [DetectionObject(x=30, y=20, width=5, height=5)],
    ]
    ids = []
    tracked_per_frame = []
    for detections in detections_per_frame:
        tracked = tracker.update(detections)
        tracked_per_frame.append(tracked)
        assert len(tracked) == 1
        ids.append(tracked[0].id)
    # Uncomment to visualize:
    if request.config.getoption("--visualize-tracking"):
        visualize_tracking(detections_per_frame, tracked_per_frame, window_name=request.node.name)
    assert len(set(ids)) == 1


def test_tracker_handles_multiple_objects(request):
    """
    Test that the tracker assigns different IDs to two objects moving in parallel.
    """
    tracker = EuclideanDistTracker()
    detections_per_frame = [
        [DetectionObject(x=10, y=10, width=5, height=5), DetectionObject(x=100, y=100, width=5, height=5)],
        [DetectionObject(x=15, y=12, width=5, height=5), DetectionObject(x=105, y=102, width=5, height=5)],
        [DetectionObject(x=20, y=15, width=5, height=5), DetectionObject(x=110, y=105, width=5, height=5)],
    ]
    ids_per_frame = []
    tracked_per_frame = []
    for detections in detections_per_frame:
        tracked = tracker.update(detections)
        tracked_per_frame.append(tracked)
        assert len(tracked) == 2
        ids_per_frame.append([obj.id for obj in tracked])
    # Uncomment to visualize:
    if request.config.getoption("--visualize-tracking"):
        visualize_tracking(detections_per_frame, tracked_per_frame, window_name=request.node.name)
    for i in range(2):
        id_sequence = [frame_ids[i] for frame_ids in ids_per_frame]
        assert len(set(id_sequence)) == 1


def test_tracker_new_object_gets_new_id(request):
    """
    Test that a new object entering the frame gets a new ID.
    """
    tracker = EuclideanDistTracker()
    detections_per_frame = [
        [DetectionObject(x=10, y=10, width=5, height=5)],
        [DetectionObject(x=15, y=12, width=5, height=5)],
        [DetectionObject(x=20, y=15, width=5, height=5), DetectionObject(x=100, y=100, width=5, height=5)],
    ]
    all_ids = set()
    tracked_per_frame = []
    for detections in detections_per_frame:
        tracked = tracker.update(detections)
        tracked_per_frame.append(tracked)
        for obj in tracked:
            all_ids.add(obj.id)
    # Uncomment to visualize:
    if request.config.getoption("--visualize-tracking"):
        visualize_tracking(detections_per_frame, tracked_per_frame, window_name=request.node.name)
    assert len(all_ids) == 2


def test_tracker_crossing_objects(request):
    """
    Test that the tracker does not swap IDs when two objects cross paths.
    """
    tracker = EuclideanDistTracker()
    detections_per_frame = [
        [DetectionObject(x=10, y=10, width=5, height=5), DetectionObject(x=100, y=100, width=5, height=5)],
        [DetectionObject(x=30, y=30, width=5, height=5), DetectionObject(x=80, y=80, width=5, height=5)],
        [DetectionObject(x=50, y=50, width=5, height=5), DetectionObject(x=60, y=60, width=5, height=5)],
        [DetectionObject(x=80, y=80, width=5, height=5), DetectionObject(x=30, y=30, width=5, height=5)],
        [DetectionObject(x=100, y=100, width=5, height=5), DetectionObject(x=10, y=10, width=5, height=5)],
    ]
    ids_per_frame = []
    tracked_per_frame = []
    for detections in detections_per_frame:
        tracked = tracker.update(detections)
        tracked_per_frame.append(tracked)
        assert len(tracked) == 2
        ids_per_frame.append([obj.id for obj in tracked])
    if request.config.getoption("--visualize-tracking"):
        visualize_tracking(detections_per_frame, tracked_per_frame, window_name=request.node.name)
    # The IDs for each object should be consistent across frames (no swap)
    for i in range(2):
        id_sequence = [frame_ids[i] for frame_ids in ids_per_frame]
        assert len(set(id_sequence)) == 1


def test_tracker_object_leaves_and_reenters(request):
    """
    Test that an object leaving and re-entering the frame gets a new ID.
    """
    tracker = EuclideanDistTracker()
    detections_per_frame = [
        [DetectionObject(x=10, y=10, width=5, height=5)],
        [],  # Object leaves
        [DetectionObject(x=12, y=12, width=5, height=5)],  # Re-enters (should get new ID)
    ]
    all_ids = set()
    tracked_per_frame = []
    for detections in detections_per_frame:
        tracked = tracker.update(detections)
        tracked_per_frame.append(tracked)
        for obj in tracked:
            all_ids.add(obj.id)
    if request.config.getoption("--visualize-tracking"):
        visualize_tracking(detections_per_frame, tracked_per_frame, window_name=request.node.name)
    # There should be two unique IDs (one for each appearance)
    assert len(all_ids) == 2


def test_tracker_variable_speed(request):
    """
    Test that the tracker can handle an object moving at variable speed.
    """
    tracker = EuclideanDistTracker()
    detections_per_frame = [
        [DetectionObject(x=10, y=10, width=5, height=5)],
        [DetectionObject(x=20, y=10, width=5, height=5)],
        [DetectionObject(x=40, y=10, width=5, height=5)],  # Large jump
        [DetectionObject(x=45, y=10, width=5, height=5)],
        [DetectionObject(x=47, y=10, width=5, height=5)],
    ]
    ids = []
    tracked_per_frame = []
    for detections in detections_per_frame:
        tracked = tracker.update(detections)
        tracked_per_frame.append(tracked)
        if tracked:
            ids.append(tracked[0].id)
    if request.config.getoption("--visualize-tracking"):
        visualize_tracking(detections_per_frame, tracked_per_frame, window_name=request.node.name)
    # All IDs should be the same (no false new object)
    assert len(set(ids)) == 1
