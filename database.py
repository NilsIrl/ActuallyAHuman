import json
import sqlite3
import time

class CVDatabase:
    def __init__(self, db_path="cv_data.db"):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.create_table()

    def create_table(self):
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS frame_data (
            timestamp REAL,
            detections TEXT,
            scene_analysis TEXT,
            movement_command TEXT
        )
        """)
        self.conn.commit()

    def insert_frame_data(self, timestamp, detections, scene_analysis, movement_command):
        # Convert YOLOv8 results to a JSON-serializable format
        detections_json = json.dumps(self.yolo_results_to_dict(detections))

        self.cursor.execute("INSERT INTO frame_data VALUES (?, ?, ?, ?)", 
                            (timestamp, detections_json, scene_analysis, movement_command))
        self.conn.commit()

    def yolo_results_to_dict(self, results):
        """ Convert YOLOv8 results to a JSON-serializable dictionary. """
        detection_list = []
        for result in results:
            for box in result.boxes:
                detection_list.append({
                    "class": int(box.cls[0].item()),  # Class index
                    "confidence": float(box.conf[0].item()),  # Confidence score
                    "bbox": [float(x) for x in box.xyxy[0].tolist()]  # Bounding box coordinates
                })
        return detection_list

    def close(self):
        self.conn.close()