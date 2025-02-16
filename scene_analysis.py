# scene_analysis.py
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import cv2

class SceneAnalyzer:
    def __init__(self, device=None):
        self.device = device or ("mps" if torch.backends.mps.is_available() 
                                   else "cuda" if torch.cuda.is_available() 
                                   else "cpu")
        # Use a GPT-based image captioning model
        self.model_name = "nlpconnect/vit-gpt2-image-captioning"
        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name).to(self.device)
        self.feature_extractor = ViTImageProcessor.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
    def analyze(self, image):
        """
        Converts an image (BGR numpy array) to a scene caption using GPT Vision.
        """
        # Convert BGR to RGB and then to PIL Image
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        pixel_values = self.feature_extractor(images=pil_image, return_tensors="pt").pixel_values.to(self.device)
        output_ids = self.model.generate(pixel_values, max_length=32, num_beams=4)
        caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return caption