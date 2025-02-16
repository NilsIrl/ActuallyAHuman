from typing import Optional
import cv2
import base64
import numpy as np
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def add_grid_to_image(frame: np.ndarray, rows: int = 10, cols: int = 10) -> np.ndarray:
    """Add numbered grid overlay to image"""
    logging.info(f"[add_grid_to_image]: Adding grid with {rows}x{cols} cells")
    if frame is None:
        logging.error("[add_grid_to_image]: Input frame is None")
        raise ValueError("Input frame is None")
        
    # Make a copy of the frame
    output = frame.copy()
    height, width = frame.shape[:2]
    
    # Calculate cell dimensions
    cell_height = height // rows
    cell_width = width // cols
    
    # Draw vertical lines
    for i in range(cols + 1):
        x = i * cell_width
        cv2.line(output, (x, 0), (x, height), (0, 255, 0), 1)
    
    # Draw horizontal lines
    for i in range(rows + 1):
        y = i * cell_height
        cv2.line(output, (0, y), (width, y), (0, 255, 0), 1)
    
    # Add numbers to cells
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    
    for i in range(rows):
        for j in range(cols):
            cell_num = i * cols + j + 1
            x = j * cell_width + cell_width // 2 - 10
            y = i * cell_height + cell_height // 2 + 5
            
            # Add black background for better visibility
            cv2.putText(output, str(cell_num), (x, y), font, font_scale, (0, 0, 0), 3)
            # Add white text
            cv2.putText(output, str(cell_num), (x, y), font, font_scale, (0, 255, 0), 1)
    
    logging.info("[add_grid_to_image]: Grid added successfully")
    return output

def compress_image(image: np.ndarray, target_resolution: tuple[int, int] = (720, 480)) -> np.ndarray:
    """Compress image to target resolution"""
    logging.info(f"[compress_image]: Compressing to {target_resolution}")
    try:
        return cv2.resize(image, target_resolution)
    except Exception as e:
        logging.error(f"[compress_image]: Error compressing image - {e}")
        raise

def take_screenshot(compression: str = '720') -> Optional[str]:
    """Take a screenshot and return it as base64 string"""
    logging.info(f"[take_screenshot]: Taking screenshot with {compression}p compression")
    if compression not in ['480', '720', '1080']:
        logging.error(f"[take_screenshot]: Invalid compression value - {compression}")
        raise ValueError("compression must be one of: '480', '720', '1080'")
    """Take a screenshot and return it as base64 string"""
    try:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            logging.error("[take_screenshot]: Failed to capture frame")
            return None
        
        if compression == '480':
            frame = compress_image(frame, (640, 480))
        elif compression == '720':
            frame = compress_image(frame, (1280, 720))
        elif compression == '1080':
            frame = compress_image(frame, (1920, 1080))
        
        # conver to rgb
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Add grid overlay
        frame_with_grid = add_grid_to_image(frame)
        
        # show the image
        plt.imshow(frame_with_grid)
        plt.show()
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', frame_with_grid)
        
        screenshot = base64.b64encode(buffer).decode('utf-8')
        
        logging.info("[take_screenshot]: Screenshot captured and processed successfully")
        return screenshot
    except Exception as e:
        logging.error(f"[take_screenshot]: Error capturing screenshot - {e}")
        return None