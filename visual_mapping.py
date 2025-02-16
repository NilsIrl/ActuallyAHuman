import cv2
import numpy as np
import math

class FloorplanMapper:
    def __init__(self, map_width, map_height, scale=100):
        """
        map_width, map_height: dimensions in pixels for the floorplan image.
        scale: pixels per meter.
        """
        self.map_width = map_width
        self.map_height = map_height
        self.scale = scale
        # Robot pose: (x, y) in meters and heading theta in radians.
        self.pose = [0.0, 0.0, 0.0]
        self.landmarks = []  # Each landmark: (global_x, global_y, label)

    def update_pose(self, delta_distance, delta_theta):
        # Update the robot pose based on encoder data.
        x, y, theta = self.pose
        theta += delta_theta
        x += delta_distance * math.cos(theta)
        y += delta_distance * math.sin(theta)
        self.pose = [x, y, theta]

    def add_landmark(self, local_x, local_y, label):
        # Convert a landmark from the robot coordinate frame to global coordinates.
        x, y, theta = self.pose
        # For a point (local_x, local_y) relative to the robot,
        # the global coordinate is:
        global_x = x + (local_x * math.cos(theta) - local_y * math.sin(theta))
        global_y = y + (local_x * math.sin(theta) + local_y * math.cos(theta))
        self.landmarks.append((global_x, global_y, label))

    def generate_map(self):
        # Create a blank floorplan image.
        map_img = np.ones((self.map_height, self.map_width, 3), dtype=np.uint8) * 255
        # For simplicity, assume the robot's origin (0,0) is at the center.
        origin = (self.map_width // 2, self.map_height // 2)
        # Draw robot trajectory (for now, just the current pose).
        x, y, theta = self.pose
        rx = int(origin[0] + x * self.scale)
        ry = int(origin[1] - y * self.scale)
        cv2.circle(map_img, (rx, ry), 5, (0, 0, 255), -1)
        arrow_end_x = int(rx + 20 * math.cos(theta))
        arrow_end_y = int(ry - 20 * math.sin(theta))
        cv2.arrowedLine(map_img, (rx, ry), (arrow_end_x, arrow_end_y), (0, 0, 255), 2)
        # Draw landmarks.
        for (lx, ly, label) in self.landmarks:
            px = int(origin[0] + lx * self.scale)
            py = int(origin[1] - ly * self.scale)
            cv2.circle(map_img, (px, py), 4, (0, 255, 0), -1)
            cv2.putText(map_img, label, (px+5, py-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        return map_img