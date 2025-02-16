import cv2
import numpy as np
import time
import math
import torch
import clip
from ultralytics import YOLO
from camera import StereoCamera
from depth_estimation import DepthEstimator
from scene_analysis import SceneAnalyzer
from convex_db import ConvexDatabase
from semantic_search import SemanticSearch
import json

# --------------------------------------------------
# Configuration
# --------------------------------------------------
TARGET_PROMPT = "grey ipad"
SIMILARITY_THRESHOLD = 0.25
MAX_BOX_AREA_FRAC = 0.30
CLOSE_THRESHOLD = 0.30
MID_THRESHOLD = 0.80

# Process only every nth frame to reduce latency
FRAME_SKIP = 2

# --------------------------------------------------
# Load Models: YOLOv8 and CLIP
# --------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
# Use a smaller YOLO model variant for faster inference.
yolo_model = YOLO("yolov8n.pt")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
with torch.no_grad():
    text_tokens = clip.tokenize([TARGET_PROMPT]).to(device)
    target_text_embedding = clip_model.encode_text(text_tokens)
    target_text_embedding /= target_text_embedding.norm(dim=-1, keepdim=True)

# --------------------------------------------------
# Initialize Modules
# --------------------------------------------------
camera = StereoCamera(left_index=0, width=1920, height=1080)
depth_estimator = DepthEstimator()
#scene_analyzer = SceneAnalyzer()
convex_db = ConvexDatabase()
semantic_search = SemanticSearch()

# --------------------------------------------------
# Canvas Layout: 2x2 grid with 16:9 cells plus scene analysis box
# Original grid: each cell was square (750x750). Now, each cell is wider.
GRID_COLS = 2
GRID_ROWS = 2
CELL_W = 750
CELL_H = int(CELL_W * 9 / 16)  # approximately 422 pixels tall
SCENE_BOX_H = 250
LEFT_CANVAS_W = GRID_COLS * CELL_W        # 1500 px
LEFT_CANVAS_H = (GRID_ROWS * CELL_H) + SCENE_BOX_H  # e.g. (2*422)+250 ≈ 1094 px

# Scale entire output window 25% bigger
FINAL_CANVAS_W = int(LEFT_CANVAS_W * 1.25)
FINAL_CANVAS_H = int(LEFT_CANVAS_H * 1.25)

frame_count = 0

# --------------------------------------------------
# Main Loop
# --------------------------------------------------
while True:
    current_time = time.time()
    left_frame, _ = camera.get_frames()
    if left_frame is None:
        print("No frame captured. Exiting...")
        break

    frame_count += 1
    # Process only every FRAME_SKIP-th frame
    if frame_count % FRAME_SKIP != 0:
        continue

    raw_frame = left_frame.copy()
    frame_h, frame_w, _ = left_frame.shape
    frame_area = frame_w * frame_h

    # --- Scene Analysis using GPT-4 Vision mini ---
    #scene_desc = scene_analyzer.analyze(left_frame)

    # --- Object Detection using YOLOv8 and CLIP filtering (only one target per frame) ---
    detection_frame = left_frame.copy()
    classification_frame = left_frame.copy()
    best_similarity = 0.0
    best_target_box = None
    combined_detections = []
    yolo_results = yolo_model(left_frame)
    for result in yolo_results:
        for box in result.boxes:
            coords = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, coords)
            conf = float(box.conf[0])
            box_area = (x2 - x1) * (y2 - y1)
            if box_area > MAX_BOX_AREA_FRAC * frame_area:
                continue
            object_name = yolo_model.names.get(int(box.cls[0].item()), "Obj")
            detection_dict = {
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class": object_name,
                "tags": [object_name.lower()]
            }
            combined_detections.append(detection_dict)
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
            cv2.putText(detection_frame, label_text, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(classification_frame, object_name, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 3)
            if similarity > SIMILARITY_THRESHOLD and similarity > best_similarity:
                best_similarity = similarity
                best_target_box = (x1, y1, x2, y2)
    target_found = best_target_box is not None
    if target_found:
        target_box = best_target_box
        tx1, ty1, tx2, ty2 = target_box
        cv2.rectangle(detection_frame, (tx1, ty1), (tx2, ty2), (0,0,255), 3)
        cv2.putText(detection_frame, f"X: {object_name} {best_similarity:.2f}", (tx1, ty1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    # --- Depth Estimation using DPT ---
    depth_map_vis, depth_map = depth_estimator.estimate_depth(left_frame)
    d_min, d_max = depth_map.min(), depth_map.max()
    norm_depth_map = (depth_map - d_min) / (d_max - d_min + 1e-6)

    # --- Classification Feed: Overlay depth labels ---
    for det in combined_detections:
        x1, y1, x2, y2 = det["bbox"]
        if x2 <= x1 or y2 <= y1:
            continue
        roi = norm_depth_map[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        avg_depth = roi.mean()
        depth_label = "Close" if avg_depth < CLOSE_THRESHOLD else "Mid" if avg_depth < MID_THRESHOLD else "Far"
        label = f"{det['class']}: {depth_label}"
        cv2.rectangle(classification_frame, (x1, y1), (x2, y2), (255,0,0), 3)
        cv2.putText(classification_frame, label, (x1, y2-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,0,0), 3)

    # --- Movement Command Decision ---
    if target_found and target_box is not None:
        tx1, ty1, tx2, ty2 = target_box
        roi_target = norm_depth_map[ty1:ty2, tx1:tx2]
        avg_depth_target = roi_target.mean() if roi_target.size > 0 else 1.0
        movement_cmd = "STOP" if avg_depth_target < CLOSE_THRESHOLD else "Straight"
    else:
        movement_cmd = "Turn Right 10°"

    # --- Build Scene Analysis Box ---
    scene_box = np.ones((SCENE_BOX_H, LEFT_CANVAS_W, 3), dtype=np.uint8) * 255
    scene_lines = [
        #f"Scene: {scene_desc[:120]}...",
        f"Target Seen: {'Yes' if target_found else 'No'}",
        f"Command: {movement_cmd}",
        f"Prompt: {TARGET_PROMPT}"
    ]
    y_pos = 40
    for line in scene_lines:
        cv2.putText(scene_box, line, (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        y_pos += 50

    # --- Assemble 2x2 Grid ---
    grid_img = np.ones((GRID_ROWS * CELL_H, LEFT_CANVAS_W, 3), dtype=np.uint8) * 255
    raw_resized = cv2.resize(raw_frame, (CELL_W, CELL_H))
    det_resized = cv2.resize(detection_frame, (CELL_W, CELL_H))
    depth_resized = cv2.resize(depth_map_vis, (CELL_W, CELL_H))
    class_resized = cv2.resize(classification_frame, (CELL_W, CELL_H))
    grid_img[0:CELL_H, 0:CELL_W] = raw_resized
    grid_img[0:CELL_H, CELL_W:LEFT_CANVAS_W] = det_resized
    grid_img[CELL_H:2*CELL_H, 0:CELL_W] = depth_resized
    grid_img[CELL_H:2*CELL_H, CELL_W:LEFT_CANVAS_W] = class_resized

    left_canvas = np.ones((LEFT_CANVAS_H, LEFT_CANVAS_W, 3), dtype=np.uint8) * 255
    left_canvas[0:GRID_ROWS * CELL_H, :] = grid_img
    left_canvas[GRID_ROWS * CELL_H:LEFT_CANVAS_H, :] = scene_box

    # --- Scale final output 25% bigger ---
    final_canvas = cv2.resize(left_canvas, (FINAL_CANVAS_W, FINAL_CANVAS_H))

    #convex_db.insert_frame_data(time.time(), combined_detections, scene_desc, movement_cmd, mapping_info={})

    cv2.imshow("CV Pipeline", final_canvas)
    if cv2.waitKey(1) == 27:
        break

camera.release()
cv2.destroyAllWindows()
convex_db.close()