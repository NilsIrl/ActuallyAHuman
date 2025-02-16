import cv2
import numpy as np

class StereoCamera:
    def __init__(self, left_index=0, width=1920, height=1080):
        """
        Assumes the stereo camera outputs a side-by-side frame.
        Only the left half is used.
        """
        self.cap = cv2.VideoCapture(left_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
    def get_frames(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None
        # Extract only the left half of the side-by-side frame.
        h, w, _ = frame.shape
        left_frame = frame[:, :w//2]
        return left_frame, None

    def rectify(self, frame):
        # Dummy rectification – replace with calibration if needed.
        return frame

    def release(self):
        self.cap.release()