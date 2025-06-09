# pip install ultralytics opencv-python deep-sort-realtime
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

# Configuration
VIDEO_PATH = "/Users/tom/Code/Bat-O-Meter/videos/batInColour.mp4"
OUTPUT_PATH = "output/bat_tracked.mp4"
MODEL_PATH = "/Users/tom/Code/Bat-O-Meter/yolov7/yolov7-tiny.pt"
CONFIDENCE_THRESHOLD = 0.3
DEVICE = "cpu"  # or 'cuda'

# Load YOLOv8 model
model = YOLO(MODEL_PATH)

# Initialize Deep SORT tracker
tracker = DeepSort(max_age=30)

# Load video
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter.fourcc(*"mp4v"), fps, (w, h))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model.predict(source=frame, device=DEVICE, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        if conf >= CONFIDENCE_THRESHOLD:
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

    # Deep SORT expects: [ [x, y, w, h], confidence, class_id ]
    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw tracks
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, w_box, h_box = track.to_ltrb()
        x1, y1, x2, y2 = map(int, [l, t, l + w_box, t + h_box])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 255, 50), 2)
        cv2.putText(frame, track_id, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 50), 2)

    out.write(frame)
    cv2.imshow("Bat Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
