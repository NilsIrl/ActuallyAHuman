import cv2
import numpy as np
import torch
import clip  # OpenAI’s CLIP package
from ultralytics import YOLO
import math

global direction_smooth, distance_smooth

# ====================================================
# 1. User Settings and Camera/Stereo Parameters
# ====================================================
TARGET_PROMPT = "pocari sweat bottle"

# P-controller gains
KP_ANGULAR = 0.001      # gentle turning
KP_LINEAR  = 0.005      # gentle forward/back motion

# Desired distance from target (in inches)
TARGET_DISTANCE_IN = 14.0

# Maximum measurable distance (in inches)
DISTANCE_MAX_IN = 120.0

# Semantic matching threshold
SIMILARITY_THRESHOLD = 0.3

# Composite Canvas configuration (2 rows x 3 columns)
SUB_W = 600  # cell width
SUB_H = 400  # cell height
CANVAS_W = 3 * SUB_W  # 1800 pixels wide
CANVAS_H = 2 * SUB_H  # 800 pixels tall

# Smoothing factor for motion parameters (exponential moving average)
ALPHA = 0.85
direction_smooth = 0.0
distance_smooth = 0.0

# ====================================================
# 2. Model Loading (YOLOv8 and CLIP)
# ====================================================
yolo_model = YOLO("yolov8m.pt")
model = YOLO("yolov8m.pt")  # for class names

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

device = "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
with torch.no_grad():
    target_tokens = clip.tokenize([TARGET_PROMPT]).to(device)
    target_embedding = clip_model.encode_text(target_tokens)
    target_embedding /= target_embedding.norm(dim=-1, keepdim=True)

# ====================================================
# 3. Stereo Camera Setup & Calibration
# ====================================================
cap = cv2.VideoCapture(0)  # expects side-by-side output from stereo camera
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Create StereoSGBM for disparity computation (tune parameters as needed)
stereo = cv2.StereoSGBM_create(
    numDisparities=24,
    blockSize=15,
    P1=8 * 3 * 15 ** 2,
    P2=32 * 3 * 15 ** 2
)

# Load calibration parameters (maps and Q matrix)
calib = np.load("stereo_calibration.npz")
left_map1  = calib["left_map1"]
left_map2  = calib["left_map2"]
right_map1 = calib["right_map1"]
right_map2 = calib["right_map2"]
Q          = calib["Q"]

# ====================================================
# 4. Helper Functions
# ====================================================
def compute_distance_from_depth_map(cx, cy, depth_map):
    """
    Given the 3D depth map (from cv2.reprojectImageTo3D) and a pixel coordinate,
    return the Z-distance in inches (assuming depth_map units are in millimeters).
    """
    depth_mm = depth_map[cy, cx, 2]
    if depth_mm <= 0 or depth_mm > 10000:
        return 9999.0
    return depth_mm / 25.4  # convert mm to inches

def compute_velocity_commands(center_x, frame_w, dist_in):
    """
    Compute linear and angular velocities using a P-controller.
    """
    image_center_x = frame_w // 2
    x_error = center_x - image_center_x
    angular_vel = -KP_ANGULAR * x_error

    z_error = dist_in - TARGET_DISTANCE_IN
    linear_vel = -KP_LINEAR * z_error

    max_linear, max_angular = 0.3, 1.0
    linear_vel = max(-max_linear, min(max_linear, linear_vel))
    angular_vel = max(-max_angular, min(max_angular, angular_vel))
    print(f"[DEBUG] Distance Error: {z_error:.2f} in")
    print(f"[DEBUG] Computed Velocities: Linear {linear_vel:.3f}, Angular {angular_vel:.3f}")
    return linear_vel, angular_vel

def send_velocity(linear, angular):
    """Placeholder to send velocity commands (currently prints them)."""
    print(f"[CMD] Linear = {linear:.3f} m/s, Angular = {angular:.3f} rad/s")

def compute_clip_similarity(image_roi):
    """Compute cosine similarity between a ROI and the target prompt using CLIP."""
    roi_rgb = cv2.cvtColor(image_roi, cv2.COLOR_BGR2RGB)
    roi_pil_jpg = cv2.imencode('.jpg', roi_rgb)[1].tobytes()
    from PIL import Image
    import io
    image_pil = Image.open(io.BytesIO(roi_pil_jpg)).convert("RGB")
    image_input = clip_preprocess(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        image_embedding = clip_model.encode_image(image_input)
        image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
        similarity = torch.nn.functional.cosine_similarity(image_embedding, target_embedding).item()
    return similarity

def draw_motion(canvas, linear, angular, smoothed_target_dist, direction_deg, raw_target_dist):
    """
    Draw an arrow and text in the motion visualization region.
    (This region is the bottom-middle cell.)
    """
    # Region coordinates for cell (bottom row, middle cell)
    region_x1 = SUB_W  # 600
    region_y1 = SUB_H  # 400
    region_w = SUB_W
    region_h = SUB_H
    center_x = region_x1 + region_w // 2
    center_y = region_y1 + region_h // 2

    arrow_len = int(100 * linear)
    arrow_angle_rad = math.radians(direction_deg)
    dx = int(arrow_len * math.cos(arrow_angle_rad))
    dy = int(arrow_len * math.sin(arrow_angle_rad))
    end_x = center_x + dx
    end_y = center_y + dy
    cv2.arrowedLine(canvas, (center_x, center_y), (end_x, end_y), (0, 0, 255), 5, tipLength=0.3)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    text_x = center_x - 100
    text_y = center_y - 100
    cv2.putText(canvas, "Robot Motion", (text_x, text_y - 40), font, 0.9, (0, 0, 255), thickness)
    cv2.putText(canvas, f"Linear: {linear:.2f} m/s", (text_x, text_y), font, scale, (0, 0, 0), thickness)
    cv2.putText(canvas, f"Angular: {angular:.2f} rad/s", (text_x, text_y + 40), font, scale, (0, 0, 0), thickness)
    cv2.putText(canvas, f"Target Dist: {raw_target_dist:.1f} in", (text_x, text_y + 80), font, scale, (0, 0, 0), thickness)
    cv2.putText(canvas, f"Angle: {direction_deg:.1f} deg", (text_x, text_y + 120), font, scale, (0, 0, 0), thickness)

# ====================================================
# 5. Main Loop: Process Video Frames
# ====================================================
while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame captured.")
        break

    # Assume the stereo camera outputs a side-by-side frame.
    h, w, _ = frame.shape
    half_w = w // 2
    left_raw = frame[:, :half_w]
    right_raw = frame[:, half_w:]

    # Rectify images using the calibration maps.
    left_rect = cv2.remap(left_raw, left_map1, left_map2, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right_raw, right_map1, right_map2, cv2.INTER_LINEAR)

    # Compute disparity on the rectified images.
    left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
    disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
    disp_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_map = cv2.reprojectImageTo3D(disparity, Q)

    # Run YOLO on the rectified left image.
    yolo_results = yolo_model(left_rect)
    bbox_img = left_rect.copy()

    # --- Target Detection (ignore obstacle branch) ---
    best_similarity = -1.0
    best_target_dist = None
    best_target_center = None

    for result in yolo_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            # Draw a green box initially.
            cv2.rectangle(bbox_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(bbox_img, f"{model.names[class_id]} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            roi = left_rect[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            similarity = compute_clip_similarity(roi)
            if similarity > SIMILARITY_THRESHOLD:
                cv2.rectangle(bbox_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(bbox_img, f"TARGET {similarity:.2f}", (x1, y1 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                dist_in = compute_distance_from_depth_map(cx, cy, depth_map)
                if dist_in > DISTANCE_MAX_IN:
                    dist_in = DISTANCE_MAX_IN
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_target_dist = dist_in
                    best_target_center = (cx, cy)

    # Compute motion commands using the detected target.
    if best_target_center is not None:
        cx, _ = best_target_center
        lin_vel, ang_vel = compute_velocity_commands(cx, bbox_img.shape[1], best_target_dist)
        send_velocity(lin_vel, ang_vel)
        # Convert angular velocity to degrees (no extra scaling factor)
        direction_raw_deg = ang_vel * 57.2958
        direction_smooth = ALPHA * direction_smooth + (1 - ALPHA) * direction_raw_deg
        distance_smooth = ALPHA * distance_smooth + (1 - ALPHA) * best_target_dist
    else:
        lin_vel, ang_vel = 0.2, 0.0
        send_velocity(lin_vel, ang_vel)
        direction_smooth = ALPHA * direction_smooth
        distance_smooth = ALPHA * distance_smooth

    # --------------------------------------------------
    # Build Composite Canvas with 6 Regions (2 rows x 3 cols)
    canvas = np.ones((CANVAS_H, CANVAS_W, 3), dtype=np.uint8) * 255

    # Region 1 (top-left): Raw Video (left_raw)
    raw_resized = cv2.resize(left_raw, (SUB_W, SUB_H))
    canvas[0:SUB_H, 0:SUB_W] = raw_resized
    cv2.putText(canvas, "Raw Video", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Region 2 (top-middle): Rectified Feed with BBoxes (bbox_img)
    bbox_resized = cv2.resize(bbox_img, (SUB_W, SUB_H))
    canvas[0:SUB_H, SUB_W:2*SUB_W] = bbox_resized
    cv2.putText(canvas, "BBoxes (Rectified)", (SUB_W+20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Region 3 (top-right): Clean Rectified Feed (left_rect)
    rect_resized = cv2.resize(left_rect, (SUB_W, SUB_H))
    canvas[0:SUB_H, 2*SUB_W:3*SUB_W] = rect_resized
    cv2.putText(canvas, "Rectified Feed", (2*SUB_W+20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Region 4 (bottom-left): Disparity Map
    disp_color = cv2.cvtColor(disp_norm, cv2.COLOR_GRAY2BGR)
    disp_resized = cv2.resize(disp_color, (SUB_W, SUB_H))
    canvas[SUB_H:2*SUB_H, 0:SUB_W] = disp_resized
    cv2.putText(canvas, "Disparity Map", (20, SUB_H+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Region 5 (bottom-middle): Motion Visualization
    motion_region = np.ones((SUB_H, SUB_W, 3), dtype=np.uint8) * 255
    draw_motion(motion_region, lin_vel, ang_vel, distance_smooth, direction_smooth,
                best_target_dist if best_target_dist is not None else 0)
    canvas[SUB_H:2*SUB_H, SUB_W:2*SUB_W] = motion_region

    # Region 6 (bottom-right): Info Region (show target distance and angle)
    info_region = np.ones((SUB_H, SUB_W, 3), dtype=np.uint8) * 255
    info_text = f"Target: {best_target_dist:.1f} in" if best_target_dist is not None else "Target: N/A"
    cv2.putText(info_region, info_text, (20, SUB_H//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(info_region, f"Angle: {direction_smooth:.1f} deg", (20, SUB_H//2 + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    canvas[SUB_H:2*SUB_H, 2*SUB_W:3*SUB_W] = info_region

    cv2.imshow("Robot Vision & Motion", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()