import cv2
import numpy as np
import torch
import clip
import math
from ultralytics import YOLO

# --------------------------------------------------
# 1. Configuration Parameters
# --------------------------------------------------
# Text prompt provided by the user
TARGET_PROMPT = "red iPad with monkey on display"

# For semantic matching, set a similarity threshold (0 to 1)
SIMILARITY_THRESHOLD = 0.25  # Tune this value as needed

# YOLOv8 target detection: we still run YOLOv8 to get candidate objects.
# All candidates will be evaluated semantically.
# (You can later refine this pipeline by skipping YOLO altogether if desired.)

# Stereo camera calibration parameters (in inches)
BASELINE_IN = 2.375       # measured distance between cameras
FOCAL_PX = 700.0          # estimated focal length in pixels; calibrate for best results

# Desired distance from target (in inches)
TARGET_DISTANCE_IN = 14.0

# P-controller gains (tuned for gentle commands)
KP_ANGULAR = 0.001
KP_LINEAR  = 0.005

# Maximum measurable distance (in inches)
DISTANCE_MAX_IN = 120.0

# Smoothing factor for motion parameters
ALPHA = 0.85  # for exponential moving average

# Display canvas configuration
CANVAS_H = 800
CANVAS_W = 1200
SUB_W = 600
SUB_H = 400

# --------------------------------------------------
# 2. Setup Models
# --------------------------------------------------
# Load YOLOv8 model from Ultralytics
yolo_model = YOLO("yolov8n.pt")

# Load CLIP model (using ViT-B/32 variant); use "mps" if available, else CPU.
device = "mps" if torch.backends.mps.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# Pre-compute text embedding for the target prompt
with torch.no_grad():
    text_tokens = clip.tokenize([TARGET_PROMPT]).to(device)
    target_text_embedding = clip_model.encode_text(text_tokens)
    target_text_embedding /= target_text_embedding.norm(dim=-1, keepdim=True)

# --------------------------------------------------
# 3. Stereo and Video Setup
# --------------------------------------------------
cap = cv2.VideoCapture(0)  # Assumes your stereo camera outputs side-by-side frames
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Setup StereoSGBM for disparity estimation
stereo = cv2.StereoSGBM_create(
    numDisparities=16,
    blockSize=15,
    P1=8 * 3 * 15 ** 2,
    P2=32 * 3 * 15 ** 2
)

# Initialize smoothed motion variables
direction_smooth = 0.0
distance_smooth = 0.0

# --------------------------------------------------
# 4. Helper Functions
# --------------------------------------------------
def triangulate_distance_in(disparity_val):
    """
    Compute distance in inches using Z = (FOCAL_PX * BASELINE_IN) / disparity_val.
    Return a large value if disparity_val is invalid.
    """
    if disparity_val <= 0:
        return 9999.0
    return (FOCAL_PX * BASELINE_IN) / float(disparity_val)

def compute_velocity_commands(center_x, frame_w, dist_in):
    """
    Compute simple P-controller commands based on horizontal error and distance error.
    Returns (linear_vel, angular_vel).
    """
    image_center_x = frame_w // 2
    x_error = center_x - image_center_x
    angular_vel = -KP_ANGULAR * x_error

    z_error = dist_in - TARGET_DISTANCE_IN
    linear_vel = -KP_LINEAR * z_error

    max_linear, max_angular = 0.3, 1.0
    linear_vel  = max(-max_linear, min(max_linear, linear_vel))
    angular_vel = max(-max_angular, min(max_angular, angular_vel))
    return linear_vel, angular_vel

def send_velocity(linear, angular):
    """
    Placeholder: send velocity commands to robot.
    Currently just prints them.
    """
    print(f"[CMD] Linear={linear:.3f}, Angular={angular:.3f}")

def draw_motion(canvas, linear, angular, distance_in, direction_deg):
    """
    Draw an arrow and text in the motion region (bottom-right).
    """
    region_x1, region_y1 = 600, 400
    region_w, region_h = SUB_W, SUB_H
    center_x = region_x1 + region_w // 2
    center_y = region_y1 + region_h // 2
    arrow_len = int(100 * linear)  # scale arrow length by linear speed
    arrow_angle_rad = math.radians(direction_deg)
    dx = int(arrow_len * math.cos(arrow_angle_rad))
    dy = int(arrow_len * math.sin(arrow_angle_rad))
    end_x = center_x + dx
    end_y = center_y + dy
    cv2.arrowedLine(canvas, (center_x, center_y), (end_x, end_y), (0,0,255), 5, tipLength=0.3)
    font, scale, color, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2
    text_x = center_x - 100
    text_y = center_y - 100
    cv2.putText(canvas, "Robot Motion", (text_x, text_y - 40), font, 0.9, (0,0,255), thickness)
    cv2.putText(canvas, f"Linear: {linear:.2f} m/s", (text_x, text_y), font, scale, color, thickness)
    cv2.putText(canvas, f"Angular: {angular:.2f} rad/s", (text_x, text_y+40), font, scale, color, thickness)
    cv2.putText(canvas, f"Distance: {distance_in:.1f} in", (text_x, text_y+80), font, scale, color, thickness)
    cv2.putText(canvas, f"Angle: {direction_deg:.1f} deg", (text_x, text_y+120), font, scale, color, thickness)

# --------------------------------------------------
# 5. Main Loop
# --------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame captured.")
        break

    # Assume the stereo camera produces a side-by-side frame.
    h, w, _ = frame.shape
    half_w = w // 2
    left_img = frame[:, :half_w]
    right_img = frame[:, half_w:]

    # Compute disparity
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    disparity = stereo.compute(left_gray, right_gray)
    disp_norm = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

    # Run YOLOv8 detection on left image to get candidate bounding boxes.
    yolo_results = yolo_model(left_img)
    bbox_img = left_img.copy()

    best_target_dist = None
    best_target_center = None

    # For each candidate, use CLIP to compare with text prompt.
    for result in yolo_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            # Draw bounding box for all objects in green initially.
            box_color = (0, 255, 0)
            cv2.rectangle(bbox_img, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(bbox_img, f"{model.names[int(box.cls[0])]} {conf:.2f}", 
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

            # Crop the image patch corresponding to the bounding box.
            patch = left_img[y1:y2, x1:x2]
            if patch.size == 0:
                continue

            # Preprocess patch for CLIP; use clip_preprocess.
            patch_pil = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            patch_pil = cv2.resize(patch_pil, (224,224))
            from PIL import Image
            patch_img = Image.fromarray(patch_pil)
            patch_input = clip_preprocess(patch_img).unsqueeze(0).to(device)

            with torch.no_grad():
                patch_embedding = clip_model.encode_image(patch_input)
                patch_embedding /= patch_embedding.norm(dim=-1, keepdim=True)
                similarity = (patch_embedding @ target_text_embedding.T).item()

            # If similarity is high enough, consider this as the target.
            if similarity > SIMILARITY_THRESHOLD:
                # Mark target bounding box in red.
                cv2.rectangle(bbox_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(bbox_img, f"TARGET {similarity:.2f}", (x1, y1 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

                # Compute the distance for this candidate using triangulation.
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                disp_val = float(disparity[cy, cx])
                dist_in = triangulate_distance_in(disp_val)
                if dist_in > DISTANCE_MAX_IN:
                    dist_in = DISTANCE_MAX_IN

                # Choose the nearest target among candidates.
                if best_target_dist is None or dist_in < best_target_dist:
                    best_target_dist = dist_in
                    best_target_center = (cx, cy)

    # Compute motion commands if a target is found.
    if best_target_center is not None:
        cx, _ = best_target_center
        lin_vel, ang_vel = compute_velocity_commands(cx, bbox_img.shape[1], best_target_dist)
        send_velocity(lin_vel, ang_vel)
        # Compute direction in degrees from angular velocity (scaled for display)
        direction_raw_deg = ang_vel * 57.2958 * 2
        # Smooth the motion parameters.
        direction_smooth = ALPHA * direction_smooth + (1 - ALPHA) * direction_raw_deg
        distance_smooth = ALPHA * distance_smooth + (1 - ALPHA) * best_target_dist
    else:
        # If no target, issue a forward command (and optionally obstacle avoidance).
        lin_vel, ang_vel = 0.2, 0.0
        send_velocity(lin_vel, ang_vel)
        direction_smooth = ALPHA * direction_smooth
        distance_smooth = ALPHA * distance_smooth

    # --------------------------------------------------
    # Create a single white canvas with 4 regions.
    canvas = np.ones((CANVAS_H, CANVAS_W, 3), dtype=np.uint8) * 255

    # Region 1 (top-left): Raw Video
    raw_resized = cv2.resize(left_img, (SUB_W, SUB_H))
    canvas[0:SUB_H, 0:SUB_W] = raw_resized
    cv2.putText(canvas, "Raw Video", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # Region 2 (top-right): BBoxes with recognition results
    bbox_resized = cv2.resize(bbox_img, (SUB_W, SUB_H))
    canvas[0:SUB_H, SUB_W:SUB_W*2] = bbox_resized
    cv2.putText(canvas, "BBoxes", (SUB_W+30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # Region 3 (bottom-left): Disparity Map
    disp_color = cv2.cvtColor(disp_norm, cv2.COLOR_GRAY2BGR)
    disp_resized = cv2.resize(disp_color, (SUB_W, SUB_H))
    canvas[SUB_H:SUB_H*2, 0:SUB_W] = disp_resized
    cv2.putText(canvas, "Disparity Map", (30, SUB_H+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # Region 4 (bottom-right): Motion visualization (arrow and text)
    draw_motion(canvas, lin_vel, ang_vel, distance_smooth, direction_smooth)

    cv2.imshow("Robot Vision & Motion", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()