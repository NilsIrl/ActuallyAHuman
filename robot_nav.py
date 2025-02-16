import cv2
import numpy as np
from ultralytics import YOLO
import math

global direction_smooth
global distance_smooth

# --------------------------------------------------
# 1. User & Camera Parameters
# --------------------------------------------------
TARGET_LABEL = "person"  # e.g., "cup", "car", "bicycle", etc.

# Simple P-control gains (tweak as needed)
KP_ANGULAR = 0.0025
KP_LINEAR  = 0.01

# Desired distance to target
TARGET_DISTANCE_MM = 1000
DEPTH_MAX_VALID    = 8000  # clamp max distance at 8m, for example

# Stereo camera parameters (approx!)
BASELINE_MM = 60.0     # e.g., 6 cm
FOCAL_PX    = 700.0    # approximate focal length in pixels

# Single Window Canvas
CANVAS_H = 800
CANVAS_W = 1200

# Sub-window sizes
SUB_W = 600
SUB_H = 400

# Smoothing factors
ALPHA = 0.8  # 0.8 means 80% old value, 20% new
direction_smooth = 0.0
distance_smooth  = 0.0

# --------------------------------------------------
# 2. Setup YOLO & Stereo
# --------------------------------------------------
model = YOLO("yolov8n.pt")  # YOLOv8 model

cap = cv2.VideoCapture(0)   # Single USB stereo camera with side-by-side frames
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# StereoSGBM
stereo = cv2.StereoSGBM_create(
    numDisparities=16,
    blockSize=15,
    P1=8 * 3 * 15 ** 2,
    P2=32 * 3 * 15 ** 2
)

# --------------------------------------------------
# 3. Helper Functions
# --------------------------------------------------

def triangulate_distance(disparity_val):
    """
    Compute distance in mm using Z = (FOCAL_PX * BASELINE_MM) / disparity_val
    Return a large number if disparity_val <= 0 or out of range.
    """
    if disparity_val <= 0:
        return 99999.0  # effectively "very far"
    dist_mm = (FOCAL_PX * BASELINE_MM) / float(disparity_val)
    return dist_mm

def compute_velocity_commands(center_x, center_y, depth_mm, frame_w, frame_h):
    """
    A simple P-controller to compute (linear_vel, angular_vel) from pixel offsets.
    """
    image_center_x = frame_w // 2
    x_error = center_x - image_center_x
    angular_vel = -KP_ANGULAR * x_error

    # forward/back offset => linear velocity
    z_error = depth_mm - TARGET_DISTANCE_MM
    linear_vel = -KP_LINEAR * z_error

    # Clamp velocities
    max_linear  = 0.3
    max_angular = 1.0
    linear_vel  = max(-max_linear, min(max_linear, linear_vel))
    angular_vel = max(-max_angular, min(max_angular, angular_vel))

    return linear_vel, angular_vel

def send_velocity(linear, angular):
    """
    Placeholder for sending commands to a real robot.
    """
    print(f"[CMD] Linear={linear:.3f}, Angular={angular:.3f}")
    # In a real robot environment, publish to /cmd_vel or call motor driver APIs.

def draw_motion(canvas, linear, angular, distance_mm, direction_deg):
    """
    Draw in bottom-right region:
      - Arrow for direction & magnitude
      - Text for linear & angular speeds
      - Distance
      - Angle
    """
    region_x1, region_y1 = 600, 400
    region_w, region_h   = SUB_W, SUB_H

    center_x = region_x1 + region_w // 2
    center_y = region_y1 + region_h // 2

    # arrow length from linear velocity
    arrow_len = int(100 * linear)

    # Convert angle in degrees to radians for arrow
    arrow_angle_rad = math.radians(direction_deg)
    dx = int(arrow_len * math.cos(arrow_angle_rad))
    dy = int(arrow_len * math.sin(arrow_angle_rad))

    end_x = center_x + dx
    end_y = center_y + dy

    # Draw arrow
    cv2.arrowedLine(
        canvas,
        (center_x, center_y),
        (end_x, end_y),
        (0, 0, 255),
        5,
        tipLength=0.3
    )

    # Put text
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    color = (0,0,0)
    thickness = 2

    text_x = center_x - 100
    text_y = center_y - 100

    cv2.putText(canvas, "Robot Motion", (text_x, text_y - 40),
                font, 0.9, (0,0,255), thickness)
    cv2.putText(canvas, f"Linear: {linear:.2f} m/s", (text_x, text_y),
                font, scale, color, thickness)
    cv2.putText(canvas, f"Angular: {angular:.2f} rad/s", (text_x, text_y + 40),
                font, scale, color, thickness)
    cv2.putText(canvas, f"Distance: {distance_mm:.1f} mm", (text_x, text_y + 80),
                font, scale, color, thickness)
    cv2.putText(canvas, f"Angle: {direction_deg:.1f} deg", (text_x, text_y + 120),
                font, scale, color, thickness)


# --------------------------------------------------
# 4. Main Loop
# --------------------------------------------------

while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame captured.")
        break

    # Split stereo frames
    h, w, _ = frame.shape
    half_w = w // 2
    left_raw  = frame[:, :half_w]
    right_raw = frame[:, half_w:]

    # Disparity
    left_gray  = cv2.cvtColor(left_raw,  cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_raw, cv2.COLOR_BGR2GRAY)
    disparity  = stereo.compute(left_gray, right_gray)
    disp_norm  = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disp_norm  = disp_norm.astype(np.uint8)

    # YOLO detection
    results = model(left_raw)
    bbox_img = left_raw.copy()
    best_depth = None
    best_center = None

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label  = model.names[cls_id]
            conf   = float(box.conf[0])
            if label == TARGET_LABEL and conf > 0.4:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(bbox_img, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(bbox_img, f"{label} {conf:.2f}",
                            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                cx = (x1 + x2)//2
                cy = (y1 + y2)//2

                # Retrieve raw disparity at (cx, cy)
                disp_val = float(disparity[cy, cx])
                dist_mm  = triangulate_distance(disp_val)
                if dist_mm > DEPTH_MAX_VALID:
                    dist_mm = DEPTH_MAX_VALID

                cv2.putText(bbox_img, f"{int(dist_mm)}mm", (cx, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                # pick nearest
                if best_depth is None or dist_mm < best_depth:
                    best_depth = dist_mm
                    best_center = (cx, cy)

    if best_center is not None:
        cx, cy = best_center
        lin_vel, ang_vel = compute_velocity_commands(cx, cy, best_depth, bbox_img.shape[1], bbox_img.shape[0])
        send_velocity(lin_vel, ang_vel)

        # Convert angular_vel to approximate degrees (scaled for visibility)
        direction_raw_deg = ang_vel * 57.2958 * 2

        # --- SMOOTHING ---
        direction_smooth = ALPHA * direction_smooth + (1 - ALPHA) * direction_raw_deg
        distance_smooth  = ALPHA * distance_smooth  + (1 - ALPHA) * best_depth

    else:
        # No target found => zero out
        lin_vel, ang_vel = 0.0, 0.0
        send_velocity(0.0, 0.0)
        direction_raw_deg = 0.0

        # slowly relax to 0 if no detection
        direction_smooth = ALPHA * direction_smooth
        distance_smooth  = ALPHA * distance_smooth

    # --------------------------------------------------
    # CREATE SINGLE WHITE CANVAS
    # --------------------------------------------------
    canvas = np.ones((CANVAS_H, CANVAS_W, 3), dtype=np.uint8) * 255

    # Sub-images
    raw_resized  = cv2.resize(left_raw,  (SUB_W, SUB_H))
    bbox_resized = cv2.resize(bbox_img,  (SUB_W, SUB_H))
    disp_color   = cv2.cvtColor(disp_norm, cv2.COLOR_GRAY2BGR)
    disp_resized = cv2.resize(disp_color, (SUB_W, SUB_H))

    # top-left: raw
    canvas[0:SUB_H, 0:SUB_W] = raw_resized
    cv2.putText(canvas, "Raw Video", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # top-right: bboxes
    canvas[0:SUB_H, SUB_W:SUB_W*2] = bbox_resized
    cv2.putText(canvas, "BBoxes", (SUB_W+30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # bottom-left: disparity
    canvas[SUB_H:SUB_H*2, 0:SUB_W] = disp_resized
    cv2.putText(canvas, "Disparity Map", (30, SUB_H+30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # bottom-right: motion
    draw_motion(
        canvas,
        lin_vel,
        ang_vel,
        distance_smooth,
        direction_smooth
    )

    cv2.imshow("Robot Vision & Motion", canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()