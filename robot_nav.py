import cv2
import numpy as np
import time
from ultralytics import YOLO

# --------------------------------------
# 1. PARAMETERS & CONSTANTS
# --------------------------------------

# Robot motion gains for simple P-control
KP_ANGULAR = 0.0025  # tuning factor for horizontal offset
KP_LINEAR  = 0.01    # tuning factor for forward motion

TARGET_DISTANCE_MM = 1000  # desired distance to object in millimeters
DEPTH_MAX_VALID    = 4000  # clamp: ignore if depth is > 4m or invalid

# Re-detection / tracking parameters
MAX_MISSED_FRAMES = 10  # if we miss the object this many frames, we re-detect
CONF_THRESHOLD    = 0.5 # YOLO confidence threshold

# --------------------------------------
# 2. CAMERA & MODELS SETUP
# --------------------------------------

# YOLO model
model = YOLO("yolov8n.pt")

# For stereo depth (assuming single USB camera with side-by-side frames)
cap = cv2.VideoCapture(0)

# Configure StereoSGBM
stereo = cv2.StereoSGBM_create(
    numDisparities=16, 
    blockSize=15,
    P1=8 * 3 * 15 ** 2, 
    P2=32 * 3 * 15 ** 2
)

# --------------------------------------
# 3. TRACKING STATE
# --------------------------------------
# We'll keep track of the object with a simple bounding-box-based "tracker":
#  - track_bbox: current bounding box [x1, y1, x2, y2]
#  - track_missed_frames: how many frames we have failed to track the object

track_bbox = None
track_missed_frames = 0

def object_lost():
    """Return True if the tracker has lost the object for too many frames."""
    global track_missed_frames
    return (track_missed_frames > MAX_MISSED_FRAMES)

# --------------------------------------
# 4. MOTION CONTROL (Pseudo-Code)
# --------------------------------------

def compute_velocity_commands(center_x, center_y, depth_mm, frame_w, frame_h):
    """
    Given the object center (center_x, center_y) in the image
    and the depth in mm, compute a simple (linear_vel, angular_vel).
    
    center_x, center_y: pixel coords of object center
    depth_mm: from disparity map
    frame_w, frame_h: used to find image center
    
    Returns: (linear_vel, angular_vel)
    """
    # 1) Horizontal offset → angular velocity
    #    Positive offset => turn left or right
    image_center_x = frame_w // 2
    x_error = (center_x - image_center_x)
    angular_vel = -KP_ANGULAR * x_error  # Negative if object is to the right => turn right

    # 2) Forward/back offset → linear velocity
    #    We want the object at ~1 meter (TARGET_DISTANCE_MM).
    z_error = (depth_mm - TARGET_DISTANCE_MM)
    linear_vel = -KP_LINEAR * z_error
    
    # Optional clamp to avoid excessive speeds
    max_linear = 0.3  # m/s
    max_angular = 1.0 # rad/s
    if linear_vel > max_linear:
        linear_vel = max_linear
    if linear_vel < -max_linear:
        linear_vel = -max_linear
    if angular_vel > max_angular:
        angular_vel = max_angular
    if angular_vel < -max_angular:
        angular_vel = -max_angular

    return (linear_vel, angular_vel)

def send_velocity(linear, angular):
    """
    Pseudo-function to command your robot motors.
    For example, you might publish to ROS /cmd_vel or call motor driver APIs.
    """
    print(f"[CMD] Linear: {linear:.3f}, Angular: {angular:.3f}")
    # TODO: Implement actual commands to your robot's motors here
    # e.g.:
    # cmd = Twist()
    # cmd.linear.x = linear
    # cmd.angular.z = angular
    # pub.publish(cmd)
    pass

# --------------------------------------
# 5. MAIN LOOP
# --------------------------------------
if not cap.isOpened():
    print("Error: Could not open stereo camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No frame captured.")
        break

    # If your stereo camera is side-by-side
    h, w, _ = frame.shape
    left_img  = frame[:, :w//2]
    right_img = frame[:, w//2:]

    # Convert to grayscale for disparity
    left_gray  = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    # Compute disparity → depth
    disparity = stereo.compute(left_gray, right_gray)
    disp_norm = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disp_norm = disp_norm.astype(np.uint8)

    # -----------------------------------
    # (A) If we have a valid track, we attempt to find the object within that ROI
    #     - If found with high confidence, update track
    #     - If not found or confidence is low, increment missed_frames
    # (B) If object_lost(), run YOLO detection on the entire image
    # -----------------------------------

    if track_bbox and not object_lost():
        # Attempt local detection around the track_bbox region
        x1, y1, x2, y2 = track_bbox
        roi = left_img[y1:y2, x1:x2]
        
        # YOLO inference on the region of interest
        # (This is an optional speed optimization; we can also do full-frame YOLO.)
        roi_results = model.predict(source=roi, conf=CONF_THRESHOLD, verbose=False)  
        
        # Check if we got any detection
        if roi_results and len(roi_results[0].boxes) > 0:
            # Let's pick the highest confidence detection
            best_box = None
            best_conf = 0
            for box in roi_results[0].boxes:
                conf = box.conf[0].item()
                if conf > best_conf:
                    best_conf = conf
                    best_box = box

            if best_box is not None and best_conf > CONF_THRESHOLD:
                # Update track_bbox (offset by x1,y1 of the ROI)
                xA = x1 + int(best_box.xyxy[0][0].item())
                yA = y1 + int(best_box.xyxy[0][1].item())
                xB = x1 + int(best_box.xyxy[0][2].item())
                yB = y1 + int(best_box.xyxy[0][3].item())
                track_bbox = [xA, yA, xB, yB]
                track_missed_frames = 0
            else:
                # Missed detection
                track_missed_frames += 1

        else:
            track_missed_frames += 1

        # If lost for too many frames, reset
        if object_lost():
            track_bbox = None
            track_missed_frames = 0

    else:
        # Either we have no track yet or we've lost the object
        # => run YOLO on the entire left image
        results = model.predict(source=left_img, conf=CONF_THRESHOLD, verbose=False)

        # If we find any detection, pick the best one
        if results and len(results[0].boxes) > 0:
            best_box = None
            best_conf = 0
            for box in results[0].boxes:
                conf = box.conf[0].item()
                if conf > best_conf:
                    best_conf = conf
                    best_box = box
            if best_box is not None:
                x1 = int(best_box.xyxy[0][0].item())
                y1 = int(best_box.xyxy[0][1].item())
                x2 = int(best_box.xyxy[0][2].item())
                y2 = int(best_box.xyxy[0][3].item())
                track_bbox = [x1, y1, x2, y2]
                track_missed_frames = 0

    # -----------------------------------
    # Draw and compute motion if we have a valid track
    # -----------------------------------
    if track_bbox:
        x1, y1, x2, y2 = track_bbox
        cv2.rectangle(left_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Compute center for depth
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Retrieve depth from disparity
        depth_val = disparity[cy, cx]
        # Depth is in "disparity units" → you'd normally calibrate to get real mm
        # For demonstration, let's assume 'depth_val' is somewhat scaled in mm,
        # or you have the formula: Z = (f * B) / disparity. 
        # We'll keep it raw or scale it as a placeholder:
        # E.g., depth_mm = convert_disparity_to_mm(depth_val)

        depth_mm = float(depth_val) * 5.0  # Example scale factor, tune per calibration
        if depth_mm > DEPTH_MAX_VALID:
            depth_mm = DEPTH_MAX_VALID  # clamp invalid to max range

        # Show depth text
        cv2.putText(left_img, f"Depth: {int(depth_mm)} mm", 
                    (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        
        # Generate motion commands
        linear, angular = compute_velocity_commands(
            center_x=cx,
            center_y=cy,
            depth_mm=depth_mm,
            frame_w=left_img.shape[1],
            frame_h=left_img.shape[0]
        )
        send_velocity(linear, angular)
    else:
        # If no track, set velocity to 0
        send_velocity(0.0, 0.0)

    # -----------------------------------
    # Show frames
    # -----------------------------------
    cv2.imshow("Left Image (Detection/Tracking)", left_img)
    cv2.imshow("Disparity", disp_norm)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()