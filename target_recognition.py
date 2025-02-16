# target_recognition.py
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import numpy as np
from transformers import CLIPProcessor, CLIPModel

class TargetRecognizer:
    def __init__(self, target_prompt="ipad with red case", device=None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        # Load CLIP model and processor
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        # Precompute the text embedding for the target prompt.
        text_inputs = self.processor(text=[target_prompt], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            self.text_features = self.clip_model.get_text_features(**text_inputs)
            self.text_features = self.text_features / self.text_features.norm(p=2, dim=-1, keepdim=True)
        self.target_prompt = target_prompt
        self.threshold = 0.25  # adjust threshold as needed

    def is_target(self, image_crop):
        """
        image_crop: a region (numpy array, BGR) cropped from the frame.
        Returns (score, bool) where bool indicates whether the crop matches the target.
        """
        # Convert BGR to RGB and then to PIL Image
        rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        # Compute cosine similarity
        similarity = F.cosine_similarity(image_features, self.text_features)
        score = similarity.item()
        return score, score > self.threshold