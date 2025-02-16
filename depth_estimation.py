# depth_estimation.py
import torch
import cv2
import numpy as np

class DepthEstimator:
    def __init__(self, device=None):
        self.device = device or (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        # Load the DPT Hybrid model via Torch Hub
        self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid").to(self.device)
        self.model.eval()

        # Load transforms for the model
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

    def estimate_depth(self, image):
        # Convert BGR to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply MiDaS transform; typically returns a (1, 3, H, W) shape
        input_tensor = self.transform(rgb).to(self.device)
        print("Transform Output Shape:", input_tensor.shape)

        # Pass directly to model (no extra unsqueeze!)
        with torch.no_grad():
            prediction = self.model(input_tensor)

        # Interpolate to original image size
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=rgb.shape[:2],
            mode="bilinear",
            align_corners=False,
        ).squeeze()

        # Convert to numpy
        depth_map = prediction.cpu().numpy()

        # Normalize for visualization
        depth_min, depth_max = depth_map.min(), depth_map.max()
        if depth_max > depth_min:
            depth_map_norm = (depth_map - depth_min) / (depth_max - depth_min)
        else:
            depth_map_norm = depth_map

        depth_map_vis = (depth_map_norm * 255).astype(np.uint8)
        depth_map_vis = cv2.applyColorMap(depth_map_vis, cv2.COLORMAP_MAGMA)
        return depth_map_vis, depth_map