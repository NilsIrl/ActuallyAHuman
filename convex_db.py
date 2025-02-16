import json
import os

class ConvexDatabase:
    def __init__(self, db_path="convex_db.json"):
        self.db_path = db_path
        if os.path.exists(self.db_path):
            with open(self.db_path, "r") as f:
                self.data = json.load(f)
        else:
            self.data = []

    def insert_frame_data(self, timestamp, detections, scene_desc, movement_cmd, mapping_info):
        record = {
            "timestamp": timestamp,
            "detections": detections,  # JSON-serializable list of detection dicts
            "scene_desc": scene_desc,
            "movement_cmd": movement_cmd,
            "mapping_info": mapping_info
        }
        self.data.append(record)
        with open(self.db_path, "w") as f:
            json.dump(self.data, f, indent=2)

    def get_all_records(self):
        return self.data

    def close(self):
        pass  # No persistent connection to close.