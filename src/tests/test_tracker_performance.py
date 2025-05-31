from batometer.detectionObject import Detection, Point
from batometer.tracker import EuclideanDistTracker
from tests.visualization import visualize_tracking


def test_tracker_tracks_moving_object(request):
    """
    Test that the tracker assigns a consistent ID to a moving object across frames.
    """
    tracker = EuclideanDistTracker()
    # Simulate a moving object across 5 frames
    detections_per_frame = [
        [Detection(Point(10, 10), 5, 5)],
        [Detection(Point(15, 12), 5, 5)],
        [Detection(Point(20, 15), 5, 5)],
        [Detection(Point(25, 18), 5, 5)],
        [Detection(Point(30, 20), 5, 5)],
    ]
    ids = []
    tracked_per_frame = []
    for detections in detections_per_frame:
        tracked = list(tracker.update(set(detections)))
        tracked_per_frame.append(tracked)
        assert len(tracked) == 1
        ids.append(tracked[0].id)
    # Uncomment to visualize:
    if request.config.getoption("--debug-frames"):
        visualize_tracking(detections_per_frame, tracked_per_frame, window_name=request.node.name)
    assert len(set(ids)) == 1


def test_tracker_handles_multiple_objects(request):
    """
    Test that the tracker assigns different IDs to two objects moving in parallel.
    """
    tracker = EuclideanDistTracker()
    detections_per_frame = [
        [Detection(Point(10, 10), 5, 5), Detection(Point(100, 100), 5, 5)],
        [Detection(Point(15, 12), 5, 5), Detection(Point(105, 102), 5, 5)],
        [Detection(Point(20, 15), 5, 5), Detection(Point(110, 105), 5, 5)],
    ]
    ids_per_frame = []
    tracked_per_frame = []
    for detections in detections_per_frame:
        tracked = list(tracker.update(set(detections)))
        tracked_per_frame.append(tracked)
        assert len(tracked) == 2
        ids_per_frame.append([obj.id for obj in tracked])
    # Uncomment to visualize:
    if request.config.getoption("--debug-frames"):
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
        [Detection(Point(10, 10), 5, 5)],
        [Detection(Point(15, 12), 5, 5)],
        [Detection(Point(20, 15), 5, 5), Detection(Point(100, 100), 5, 5)],
    ]
    all_ids = set()
    tracked_per_frame = []
    for detections in detections_per_frame:
        tracked = tracker.update(set(detections))
        tracked_per_frame.append(list(tracked))
        for obj in tracked:
            all_ids.add(obj.id)
    # Uncomment to visualize:
    if request.config.getoption("--debug-frames"):
        visualize_tracking(detections_per_frame, tracked_per_frame, window_name=request.node.name)
    assert len(all_ids) == 2


def test_tracker_crossing_objects(request):
    """
    Test that the tracker does not swap IDs when two objects cross paths.
    """
    tracker = EuclideanDistTracker()
    detections_per_frame = [
        [Detection(Point(10, 10), 5, 5), Detection(Point(100, 100), 5, 5)],
        [Detection(Point(30, 30), 5, 5), Detection(Point(80, 80), 5, 5)],
        [Detection(Point(50, 50), 5, 5), Detection(Point(60, 60), 5, 5)],
        [Detection(Point(80, 80), 5, 5), Detection(Point(30, 30), 5, 5)],
        [Detection(Point(100, 100), 5, 5), Detection(Point(10, 10), 5, 5)],
    ]
    ids_per_frame = []
    tracked_per_frame = []
    for detections in detections_per_frame:
        tracked = list(tracker.update(set(detections)))
        tracked_per_frame.append(tracked)
        assert len(tracked) == 2
        ids_per_frame.append([obj.id for obj in tracked])
    if request.config.getoption("--debug-frames"):
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
        [Detection(Point(10, 10), 5, 5)],
        [],  # Object leaves
        [Detection(Point(12, 12), 5, 5)],  # Re-enters (should get new ID)
    ]
    all_ids = set()
    tracked_per_frame = []
    for detections in detections_per_frame:
        tracked = tracker.update(set(detections))
        tracked_per_frame.append(list(tracked))
        for obj in tracked:
            all_ids.add(obj.id)
    if request.config.getoption("--debug-frames"):
        visualize_tracking(detections_per_frame, tracked_per_frame, window_name=request.node.name)
    # There should be two unique IDs (one for each appearance)
    assert len(all_ids) == 2


def test_tracker_variable_speed(request):
    """
    Test that the tracker can handle an object moving at variable speed.
    """
    tracker = EuclideanDistTracker()
    detections_per_frame = [
        [Detection(Point(10, 10), 5, 5)],
        [Detection(Point(20, 10), 5, 5)],
        [Detection(Point(40, 10), 5, 5)],  # Large jump
        [Detection(Point(45, 10), 5, 5)],
        [Detection(Point(47, 10), 5, 5)],
    ]
    ids = []
    tracked_per_frame = []
    for detections in detections_per_frame:
        tracked = list(tracker.update(set(detections)))
        tracked_per_frame.append(tracked)
        if tracked:
            ids.append(tracked[0].id)
    if request.config.getoption("--debug-frames"):
        visualize_tracking(detections_per_frame, tracked_per_frame, window_name=request.node.name)
    # All IDs should be the same (no false new object)
    assert len(set(ids)) == 1


def test_tracker_empty_frame(request):
    """
    Test that the tracker handles frames with no detections (empty frame) gracefully.
    """
    tracker = EuclideanDistTracker()
    detections_per_frame = [
        [Detection(Point(10, 10), 5, 5)],
        [],  # No detections
        [Detection(Point(12, 12), 5, 5)],
    ]
    tracked_per_frame = []
    for detections in detections_per_frame:
        tracked = list(tracker.update(set(detections)))
        tracked_per_frame.append(tracked)
    if request.config.getoption("--debug-frames"):
        visualize_tracking(detections_per_frame, tracked_per_frame, window_name=request.node.name)
    # Should not crash, and IDs should be unique for each appearance
    all_ids = [obj.id for frame in tracked_per_frame for obj in frame]
    assert len(set(all_ids)) == 2


def test_tracker_overlapping_objects(request):
    """
    Test that the tracker does not merge IDs when two objects overlap and then separate.
    """
    tracker = EuclideanDistTracker()
    detections_per_frame = [
        [Detection(Point(10, 10), 10, 10), Detection(Point(100, 100), 10, 10)],
        [Detection(Point(50, 50), 20, 20), Detection(Point(55, 55), 20, 20)],
        [Detection(Point(100, 100), 10, 10), Detection(Point(10, 10), 10, 10)],
    ]
    ids_per_frame = []
    tracked_per_frame = []
    for detections in detections_per_frame:
        tracked = list(tracker.update(set(detections)))
        tracked_per_frame.append(tracked)
        ids_per_frame.append([obj.id for obj in tracked])
    if request.config.getoption("--debug-frames"):
        visualize_tracking(detections_per_frame, tracked_per_frame, window_name=request.node.name)
    # IDs should be consistent for each object
    for i in range(2):
        id_sequence = [frame_ids[i] for frame_ids in ids_per_frame if len(frame_ids) > i]
        assert len(set(id_sequence)) == 1


def test_tracker_many_objects(request):
    """
    Test that the tracker can handle a large number of objects in a single frame.
    """
    tracker = EuclideanDistTracker()
    detections_per_frame = [
        [Detection(Point(10 * i, 10 * i), 5, 5) for i in range(10)],
        [Detection(Point(10 * i + 2, 10 * i + 2), 5, 5) for i in range(10)],
        [Detection(Point(10 * i + 4, 10 * i + 4), 5, 5) for i in range(10)],
    ]
    tracked_per_frame = []
    for detections in detections_per_frame:
        tracked = list(tracker.update(set(detections)))
        tracked_per_frame.append(tracked)
        assert len(tracked) == 10
    if request.config.getoption("--debug-frames"):
        visualize_tracking(detections_per_frame, tracked_per_frame, window_name=request.node.name)
    # Each object should keep its ID
    ids_per_object = list(zip(*[[obj.id for obj in frame] for frame in tracked_per_frame]))
    for id_seq in ids_per_object:
        assert len(set(id_seq)) == 1


def test_tracker_bounding_box_change(request):
    """
    Test that the tracker maintains the same ID when the bounding box size/shape changes.
    """
    tracker = EuclideanDistTracker()
    detections_per_frame = [
        [Detection(Point(10, 10), 5, 5)],
        [Detection(Point(12, 10), 10, 8)],  # Size/shape change
        [Detection(Point(15, 12), 7, 6)],
    ]
    ids = []
    tracked_per_frame = []
    for detections in detections_per_frame:
        tracked = list(tracker.update(set(detections)))
        tracked_per_frame.append(tracked)
        if tracked:
            ids.append(tracked[0].id)
    if request.config.getoption("--debug-frames"):
        visualize_tracking(detections_per_frame, tracked_per_frame, window_name=request.node.name)
    assert len(set(ids)) == 1


def test_tracker_out_of_bounds(request):
    """
    Test that the tracker handles detections with out-of-bounds coordinates gracefully.
    """
    tracker = EuclideanDistTracker()
    detections_per_frame = [
        [Detection(Point(-10, -10), 5, 5)],  # Negative coords
        [Detection(Point(410, 410), 5, 5)],  # Outside window
        [Detection(Point(10, 10), 5, 5)],
    ]
    tracked_per_frame = []
    for detections in detections_per_frame:
        tracked = list(tracker.update(set(detections)))
        tracked_per_frame.append(tracked)
    if request.config.getoption("--debug-frames"):
        visualize_tracking(detections_per_frame, tracked_per_frame, window_name=request.node.name)
    # Should not crash, and should assign unique IDs
    all_ids = [obj.id for frame in tracked_per_frame for obj in frame]
    assert len(set(all_ids)) == 3
