import cv2
import numpy as np
import time
import torch
import clip
from ultralytics import YOLO
from camera import StereoCamera
from depth_estimation import DepthEstimator
#from scene_analysis import SceneAnalyzer  # Disabled for performance
#from convex_db import ConvexDatabase
from semantic_search import SemanticSearch
import argparse
from PIL import Image
from flask import Flask, Response
import threading

# --------------------------------------------------
# Parse Arguments
# --------------------------------------------------
parser = argparse.ArgumentParser(description='CV Pipeline with optional GUI and web streaming')
parser.add_argument('--gui', action='store_true', help='Enable local GUI display')
args = parser.parse_args()

# --------------------------------------------------
# Configuration
# --------------------------------------------------
TARGET_PROMPT = "grey ipad"
SIMILARITY_THRESHOLD = 0.25
MAX_BOX_AREA_FRAC = 0.30
CLOSE_THRESHOLD = 0.30
MID_THRESHOLD = 0.80
FRAME_SKIP = 2  # Process every nth frame

# --------------------------------------------------
# Load Models: YOLOv8 and CLIP
# --------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
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
# scene_analyzer = SceneAnalyzer()  # Disabled for performance
# convex_db = ConvexDatabase()
semantic_search = SemanticSearch()

# --------------------------------------------------
# Canvas Layout Settings
# --------------------------------------------------
# Adjust feed dimensions: each feed is 920 x 517 (approx 16:9)
GRID_COLS = 2
GRID_ROWS = 2
CELL_W = 920
CELL_H = int(CELL_W * 9 / 16)  # ~517 px
DASHBOARD_H = 350  # Taller dashboard so text isn't cut off
LEFT_CANVAS_W = GRID_COLS * CELL_W          # 2*920 = 1840 px
LEFT_CANVAS_H = (GRID_ROWS * CELL_H) + DASHBOARD_H  # (2*517)+350 = 1384 px

# We will stream the output at its native resolution (1840 x 1384)
FINAL_CANVAS_W = LEFT_CANVAS_W
FINAL_CANVAS_H = LEFT_CANVAS_H

# Global variable to hold the latest frame for streaming
output_frame = None
frame_lock = threading.Lock()

# --------------------------------------------------
# Vision Processing Loop
# --------------------------------------------------
def vision_loop():
    global output_frame
    frame_count = 0
    prev_time = time.time()

    while True:
        current_time = time.time()
        dt = current_time - prev_time
        fps = 1 / dt if dt > 0 else 0
        prev_time = current_time

        left_frame, _ = camera.get_frames()
        if left_frame is None:
            print("No frame captured. Exiting vision loop...")
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        raw_frame = left_frame.copy()
        frame_h, frame_w, _ = left_frame.shape
        frame_area = frame_w * frame_h

        # --- Scene Analysis (disabled) ---
        scene_desc = "Scene analysis disabled."

        # --- YOLO Object Detection ---
        yolo_start = time.time()
        yolo_results = yolo_model(left_frame)
        yolo_time = (time.time() - yolo_start) * 1000.0  # ms

        detection_frame = left_frame.copy()
        classification_frame = left_frame.copy()
        best_similarity = 0.0
        best_target_box = None
        combined_detections = []
        clip_patches = []   # For batch CLIP processing
        patch_info = []     # (detection index, bbox)

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
                cv2.rectangle(detection_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                patch = left_frame[y1:y2, x1:x2]
                if patch.size == 0:
                    continue
                patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                patch_rgb = cv2.resize(patch_rgb, (224, 224))
                patch_img = Image.fromarray(patch_rgb)
                clip_patches.append(patch_img)
                patch_info.append((len(combined_detections) - 1, (x1, y1, x2, y2)))

        # --- Batch CLIP Inference ---
        clip_time = 0.0
        if clip_patches:
            clip_start = time.time()
            batch_input = [clip_preprocess(img) for img in clip_patches]
            batch_input = torch.stack(batch_input).to(device)
            with torch.no_grad():
                batch_embeddings = clip_model.encode_image(batch_input)
                batch_embeddings /= batch_embeddings.norm(dim=-1, keepdim=True)
                similarities = (batch_embeddings @ target_text_embedding.T).squeeze(1).cpu().tolist()
            clip_time = (time.time() - clip_start) * 1000.0  # ms

            for idx, bbox in patch_info:
                sim = similarities.pop(0)
                combined_detections[idx]["similarity"] = sim
                x1, y1, x2, y2 = bbox
                label_text = f"{combined_detections[idx]['class']} {sim:.2f}"
                cv2.putText(detection_frame, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(classification_frame, combined_detections[idx]['class'], (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
                if sim > SIMILARITY_THRESHOLD and sim > best_similarity:
                    best_similarity = sim
                    best_target_box = bbox

        target_found = best_target_box is not None
        if target_found:
            tx1, ty1, tx2, ty2 = best_target_box
            cv2.rectangle(detection_frame, (tx1, ty1), (tx2, ty2), (0, 0, 255), 3)
            cv2.putText(detection_frame, f"Target {best_similarity:.2f}", (tx1, ty1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # --- Depth Estimation ---
        depth_map_vis, depth_map = depth_estimator.estimate_depth(left_frame)
        d_min, d_max = depth_map.min(), depth_map.max()
        norm_depth_map = (depth_map - d_min) / (d_max - d_min + 1e-6)

        # --- Overlay Depth Labels ---
        for det in combined_detections:
            x1, y1, x2, y2 = det["bbox"]
            if x2 <= x1 or y2 <= y1:
                continue
            roi = norm_depth_map[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            avg_depth = roi.mean()
            depth_label = "Close" if avg_depth < CLOSE_THRESHOLD else "Mid" if avg_depth < MID_THRESHOLD else "Far"
            det["avg_depth"] = avg_depth
            det["depth_label"] = depth_label
            label = f"{det['class']}: {depth_label}"
            cv2.rectangle(classification_frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.putText(classification_frame, label, (x1, y2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

        # --- Movement Command Decision ---
        if target_found and best_target_box is not None:
            tx1, ty1, tx2, ty2 = best_target_box
            roi_target = norm_depth_map[ty1:ty2, tx1:tx2]
            avg_depth_target = roi_target.mean() if roi_target.size > 0 else 1.0
            if avg_depth_target < CLOSE_THRESHOLD:
                target_center = (tx1 + tx2) / 2.0
                frame_center = frame_w / 2.0
                horizontal_error = target_center - frame_center
                error_threshold = 50  # pixels
                if abs(horizontal_error) < error_threshold:
                    movement_cmd = "STOP"
                elif horizontal_error > 0:
                    movement_cmd = "Turn Right"
                else:
                    movement_cmd = "Turn Left"
            else:
                movement_cmd = "Straight"
        else:
            movement_cmd = "Turn Right 10°"

        # --- Build Dashboard / Information Overlay ---
        dashboard_box = np.ones((DASHBOARD_H, LEFT_CANVAS_W, 3), dtype=np.uint8) * 255
        # Use black font for stats.
        dashboard_lines = [
            f"FPS: {fps:.1f}   Frame: {frame_count}",
            f"YOLO Time: {yolo_time:.1f}ms   CLIP Time: {clip_time:.1f}ms",
            f"Detections: {len(combined_detections)}",
            f"Target Prompt: {TARGET_PROMPT}",
            f"Command: {movement_cmd}"
        ]
        y_pos = 30
        for line in dashboard_lines:
            cv2.putText(dashboard_box, line, (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            y_pos += 40
        cv2.putText(dashboard_box, "Detections:", (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        y_pos += 30
        for det in combined_detections[:5]:
            line = (f"{det['class']} (Conf: {det['confidence']:.2f}, "
                    f"Sim: {det.get('similarity', 0):.2f}, Depth: {det.get('depth_label','N/A')})")
            cv2.putText(dashboard_box, line, (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            y_pos += 30
        if len(combined_detections) > 5:
            extra = len(combined_detections) - 5
            cv2.putText(dashboard_box, f"... and {extra} more", (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # --- Assemble the 2x2 Grid ---
        grid_img = np.ones((GRID_ROWS * CELL_H, LEFT_CANVAS_W, 3), dtype=np.uint8)
        raw_resized = cv2.resize(raw_frame, (CELL_W, CELL_H))
        det_resized = cv2.resize(detection_frame, (CELL_W, CELL_H))
        depth_resized = cv2.resize(depth_map_vis, (CELL_W, CELL_H))
        class_resized = cv2.resize(classification_frame, (CELL_W, CELL_H))
        grid_img[0:CELL_H, 0:CELL_W] = raw_resized
        grid_img[0:CELL_H, CELL_W:LEFT_CANVAS_W] = det_resized
        grid_img[CELL_H:2*CELL_H, 0:CELL_W] = depth_resized
        grid_img[CELL_H:2*CELL_H, CELL_W:LEFT_CANVAS_W] = class_resized

        # --- Assemble Full Canvas (Grid on top, Dashboard below) ---
        left_canvas = np.ones((LEFT_CANVAS_H, LEFT_CANVAS_W, 3), dtype=np.uint8)
        left_canvas[0:GRID_ROWS * CELL_H, :] = grid_img
        left_canvas[GRID_ROWS * CELL_H:LEFT_CANVAS_H, :] = dashboard_box

        # Final output (no further scaling; streaming at native resolution)
        final_canvas = left_canvas.copy()

        # Update global output frame with thread locking.
        with frame_lock:
            output_frame = final_canvas.copy()

        # Optionally also show locally if --gui flag is set.
        if args.gui:
            cv2.imshow("CV Pipeline", final_canvas)
            if cv2.waitKey(1) == 27:
                break

    camera.release()
    if args.gui:
        cv2.destroyAllWindows()

# --------------------------------------------------
# Flask Web Streaming Setup
# --------------------------------------------------
app = Flask(__name__)

def generate_frames():
    """Encode the latest frame as JPEG and stream it."""
    global output_frame
    while True:
        with frame_lock:
            if output_frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', output_frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# --------------------------------------------------
# Run Vision Loop in a Background Thread & Start Flask Server
# --------------------------------------------------
if __name__ == '__main__':
    vision_thread = threading.Thread(target=vision_loop, daemon=True)
    vision_thread.start()
    # Start Flask web server on port 5000 (accessible via http://<your-ip>:5000)
    app.run(host='0.0.0.0', port=5001, threaded=True)