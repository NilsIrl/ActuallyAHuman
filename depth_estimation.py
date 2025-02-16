# depth_estimation.py
import torch
import cv2
import numpy as np
from transformers import DPTForDepthEstimation, DPTImageProcessor

class DepthEstimator:
    def __init__(self, model_name="Intel/dpt-large"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DPTForDepthEstimation.from_pretrained(model_name).to(self.device)
        self.image_processor = DPTImageProcessor.from_pretrained(model_name)

    def estimate_depth(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = self.image_processor(images=frame_rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        depth_map = prediction.cpu().numpy()
        depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_map_colored = cv2.applyColorMap(depth_map_normalized.astype(np.uint8), cv2.COLORMAP_INFERNO)
        return depth_map_colored, depth_map