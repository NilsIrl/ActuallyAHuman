# scene_analysis.py
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import cv2

class SceneAnalyzer:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "nlpconnect/vit-gpt2-image-captioning"
        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name).to(self.device)
        self.feature_extractor = ViTImageProcessor.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
    def analyze(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        pixel_values = self.feature_extractor(images=pil_image, return_tensors="pt").pixel_values.to(self.device)
        output_ids = self.model.generate(pixel_values, max_length=32, num_beams=4)
        caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return caption