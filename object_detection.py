# object_detection.py
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path="yolov8n.pt"):
        # Load the YOLOv8 model (adjust model variant/path as needed)
        self.model = YOLO(model_path)
    
    def detect(self, image):
        """
        Runs detection on the given image.
        Returns a list of detections where each detection is a dict:
            {'box': [x1, y1, x2, y2], 'confidence': conf, 'class': class_name}
        """
        results = self.model(image)[0]
        detections = []
        for det in results.boxes.data.tolist():
            # YOLOv8 returns: [x1, y1, x2, y2, confidence, class]
            x1, y1, x2, y2, conf, cls = det
            detections.append({
                'box': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': conf,
                'class': int(cls)
            })
        return detections