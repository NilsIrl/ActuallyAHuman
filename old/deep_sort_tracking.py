import cv2
import numpy as np
import sys
sys.path.append("/Users/robertzhang/Documents/GitHub/ActuallyAHuman/deep_sort")
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.kalman_filter import KalmanFilter
from deep_sort.linear_assignment import min_cost_matching
from deep_sort.iou_matching import iou_cost
from ultralytics import YOLO
from deep_sort.deep_sort import DeepSort

# Load YOLOv8 Model
model = YOLO("yolov8n.pt")

# Load DeepSORT Tracker
metric = NearestNeighborDistanceMetric("cosine", matching_threshold=0.2, budget=100)
tracker = Tracker(metric)

# Open Stereo Camera (Assumes Side-by-Side Layout)
cap = cv2.VideoCapture(0)  # Change index if needed

# StereoBM Matcher for Depth
stereo = cv2.StereoSGBM_create(
    numDisparities=16, blockSize=15,
    P1=8 * 3 * 15 ** 2, P2=32 * 3 * 15 ** 2
)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get frame dimensions and split into left & right images
    height, width, _ = frame.shape
    left_img = frame[:, :width//2]
    right_img = frame[:, width//2:]

    # Convert to Grayscale
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    # Compute Disparity Map (Depth)
    disparity = stereo.compute(left_gray, right_gray)

    # Normalize disparity for visualization
    disp_norm = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Run YOLOv8 Detection
    results = model(left_img)

    detections = []
    confidences = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()

            # Get depth at object center
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            depth = disparity[center_y, center_x]  # Get depth value

            detections.append(Detection([x1, y1, x2, y2], conf, None))
            confidences.append(conf)

    # Update DeepSORT Tracker
    tracker.predict()
    tracker.update(detections)

    # Draw Tracking Results
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue

        x1, y1, x2, y2 = map(int, track.to_tlbr())
        track_id = track.track_id

        cv2.rectangle(left_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(left_img, f"ID: {track_id} Depth: {depth}mm", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Show results
    cv2.imshow("Stereo Tracking", left_img)
    cv2.imshow("Disparity Map", disp_norm.astype(np.uint8))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()