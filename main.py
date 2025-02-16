import cv2
import numpy as np
import time
import math
import torch
import clip
from ultralytics import YOLO
from camera import StereoCamera
from depth_estimation import DepthEstimator
from database import CVDatabase
from scene_analysis import SceneAnalyzer

# --------------------------------------------------
# Configuration Parameters
# --------------------------------------------------
TARGET_PROMPT = "vitamin water"
SIMILARITY_THRESHOLD = 0.25  # Adjust as needed
MAX_BOX_AREA_FRAC = 0.30     # Reject candidate boxes covering >30% of frame area
# Depth thresholds (normalized)
CLOSE_THRESHOLD = 0.30
MID_THRESHOLD = 0.80  # Anything >= this is "Far"

# --------------------------------------------------
# Load Models: YOLO and CLIP
# --------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
with torch.no_grad():
    text_tokens = clip.tokenize([TARGET_PROMPT]).to(device)
    target_text_embedding = clip_model.encode_text(text_tokens)
    target_text_embedding /= target_text_embedding.norm(dim=-1, keepdim=True)

yolo_model = YOLO("yolov8n.pt")

# --------------------------------------------------
# Initialize Other Modules
# --------------------------------------------------
camera = StereoCamera(left_index=0, width=1920, height=1080)
depth_estimator = DepthEstimator()
database = CVDatabase()
scene_analyzer = SceneAnalyzer()

# --------------------------------------------------
# Scene Analysis Update Timing
# --------------------------------------------------
last_scene_update = 0
scene_desc = "Analyzing..."
SCENE_UPDATE_INTERVAL = 60  # seconds

# --------------------------------------------------
# Canvas Layout Configuration
# --------------------------------------------------
# We use a 2x2 grid and a scene analysis box below.
CANVAS_W = 1500
GRID_ROWS = 2
GRID_COLS = 2
CELL_W = CANVAS_W // GRID_COLS         # 750 px per cell width
CELL_H = 600                           # 600 px per cell height
SCENE_BOX_H = 250                      # Scene analysis box height
CANVAS_H = (CELL_H * GRID_ROWS) + SCENE_BOX_H  # 1200 + 250 = 1450

# For target centering.
TARGET_THRESHOLD = 50  # pixels

while True:
    current_time = time.time()
    
    # ---------------------------
    # Capture Left Frame Only
    # ---------------------------
    left_frame, _ = camera.get_frames()
    if left_frame is None:
        print("Failed to capture frame. Exiting...")
        break
    raw_frame = left_frame.copy()
    rectified = camera.rectify(left_frame)
    frame_h, frame_w, _ = left_frame.shape
    frame_area = frame_w * frame_h

    # ---------------------------
    # Scene Analysis Update (using GPT-Vision)
    # ---------------------------
    if current_time - last_scene_update >= SCENE_UPDATE_INTERVAL:
        scene_desc = scene_analyzer.analyze(left_frame)
        last_scene_update = current_time

    # ---------------------------
    # Feed: Object Detection (YOLO+CLIP Hybrid)
    # ---------------------------
    detection_frame = left_frame.copy()
    classification_frame = left_frame.copy()
    target_found = False
    target_box = None

    yolo_results = yolo_model(left_frame)
    for result in yolo_results:
        for box in result.boxes:
            coords = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, coords)
            conf = float(box.conf[0])
            box_area = (x2 - x1) * (y2 - y1)
            if box_area > MAX_BOX_AREA_FRAC * frame_area:
                continue  # Skip overly large boxes

            # Get object name if available.
            object_name = yolo_model.names.get(int(box.cls[0].item()), "Obj")

            # Draw initial bounding box (green) on detection feed.
            cv2.rectangle(detection_frame, (x1, y1), (x2, y2), (0,255,0), 2)

            patch = left_frame[y1:y2, x1:x2]
            if patch.size == 0:
                continue
            patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            patch_rgb = cv2.resize(patch_rgb, (224,224))
            from PIL import Image
            patch_img = Image.fromarray(patch_rgb)
            patch_input = clip_preprocess(patch_img).unsqueeze(0).to(device)
            with torch.no_grad():
                patch_embedding = clip_model.encode_image(patch_input)
                patch_embedding /= patch_embedding.norm(dim=-1, keepdim=True)
                similarity = (patch_embedding @ target_text_embedding.T).item()
            label_text = f"{object_name} {similarity:.2f}"

            if similarity > SIMILARITY_THRESHOLD:
                target_found = True
                target_box = (x1, y1, x2, y2)
                cv2.rectangle(detection_frame, (x1, y1), (x2, y2), (0,0,255), 3)
                cv2.putText(detection_frame, f"X: {object_name} {similarity:.2f}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
            else:
                cv2.putText(detection_frame, label_text, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            # In the classification feed, also display the object name.
            cv2.putText(classification_frame, object_name, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 3)

    # ---------------------------
    # Feed: Depth Map
    # ---------------------------
    depth_map_vis, depth_map = depth_estimator.estimate_depth(left_frame)
    d_min, d_max = depth_map.min(), depth_map.max()
    norm_depth_map = (depth_map - d_min) / (d_max - d_min + 1e-6)

    # ---------------------------
    # Feed: Classification (with Depth Label)
    # ---------------------------
    for result in yolo_results:
        for box in result.boxes:
            coords = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, coords)
            if x2 <= x1 or y2 <= y1:
                continue
            roi = norm_depth_map[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            avg_depth = roi.mean()
            if avg_depth < CLOSE_THRESHOLD:
                depth_label = "Close"
            elif avg_depth < MID_THRESHOLD:
                depth_label = "Mid"
            else:
                depth_label = "Far"
            object_name = yolo_model.names.get(int(box.cls[0].item()), "Obj")
            label = f"{object_name}: {depth_label}"
            cv2.rectangle(classification_frame, (x1, y1), (x2, y2), (255,0,0), 3)
            cv2.putText(classification_frame, label, (x1, y2-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,0,0), 3)

    # ---------------------------
    # Determine Movement Command
    # ---------------------------
    # Recognition flow:
    # 1. If target ("X") not seen -> Command: "Turn Right 10°"
    # 2. If seen -> compute average depth of target box:
    #       If target is close -> "STOP"; else -> "Straight"
    if target_found and target_box is not None:
        tx1, ty1, tx2, ty2 = target_box
        roi_target = norm_depth_map[ty1:ty2, tx1:tx2]
        avg_depth_target = roi_target.mean() if roi_target.size > 0 else 1.0
        motion_cmd = "STOP" if avg_depth_target < CLOSE_THRESHOLD else "Straight"
    else:
        motion_cmd = "Turn Right 10°"

    # ---------------------------
    # Create Scene Analysis Box (Full-Width Rectangle)
    # ---------------------------
    scene_box = np.ones((SCENE_BOX_H, CANVAS_W, 3), dtype=np.uint8) * 255
    scene_lines = [
        f"Scene: {scene_desc[:120]}...",  # truncate if too long
        f"Target Seen: {'Yes' if target_found else 'No'}",
        f"Command: {motion_cmd}",
        f"Prompt: {TARGET_PROMPT}"
    ]
    y_pos = 40
    for line in scene_lines:
        cv2.putText(scene_box, line, (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        y_pos += 50

    # ---------------------------
    # Assemble Final Canvas
    # ---------------------------
    # Create a 2x2 grid of feeds:
    # Top-left: Raw Video
    # Top-right: Detections (Bounding Boxes)
    # Bottom-left: Depth Map
    # Bottom-right: Classification
    grid_img = np.ones((CELL_H*GRID_ROWS, CANVAS_W, 3), dtype=np.uint8) * 255
    raw_resized = cv2.resize(raw_frame, (CELL_W, CELL_H))
    det_resized = cv2.resize(detection_frame, (CELL_W, CELL_H))
    depth_resized = cv2.resize(depth_map_vis, (CELL_W, CELL_H))
    class_resized = cv2.resize(classification_frame, (CELL_W, CELL_H))
    
    # Place cells in the grid.
    grid_img[0:CELL_H, 0:CELL_W] = raw_resized
    grid_img[0:CELL_H, CELL_W:CANVAS_W] = det_resized
    grid_img[CELL_H:2*CELL_H, 0:CELL_W] = depth_resized
    grid_img[CELL_H:2*CELL_H, CELL_W:CANVAS_W] = class_resized

    # Create final canvas by stacking the grid and the scene analysis box.
    canvas = np.ones((CANVAS_H, CANVAS_W, 3), dtype=np.uint8) * 255
    canvas[0:CELL_H*GRID_ROWS, :] = grid_img
    canvas[CELL_H*GRID_ROWS:CANVAS_H, :] = scene_box

    # Add labels for each cell on the grid.
    cv2.putText(canvas, "Raw Video", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.putText(canvas, "Detections", (CELL_W+20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.putText(canvas, "Depth Map", (20, CELL_H+40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.putText(canvas, "Classification", (CELL_W+20, CELL_H+40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # Log data to the database.
    database.insert_frame_data(time.time(), yolo_results, scene_desc, motion_cmd)

    cv2.imshow("CV Pipeline", canvas)
    if cv2.waitKey(1) == 27:
        break

camera.release()
cv2.destroyAllWindows()
database.close()