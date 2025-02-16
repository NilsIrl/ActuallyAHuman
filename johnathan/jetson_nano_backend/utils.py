from typing import Optional
import uuid
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
    mapping_from_cell_num_to_location = {}
    
    for i in range(rows):
        for j in range(cols):
            cell_num = i * cols + j + 1
            x = j * cell_width + cell_width // 2 - 10
            y = i * cell_height + cell_height // 2 + 5
            
            # Add black background for better visibility
            cv2.putText(output, str(cell_num), (x, y), font, font_scale, (0, 0, 0), 3)
            # Add white text
            cv2.putText(output, str(cell_num), (x, y), font, font_scale, (0, 255, 0), 1)
            mapping_from_cell_num_to_location[cell_num] = (j * cell_width, i * cell_height, (j + 1) * cell_width, (i + 1) * cell_height)
    logging.info("[add_grid_to_image]: Grid added successfully")
    return output, mapping_from_cell_num_to_location

def compress_image(image: np.ndarray, target_resolution: tuple[int, int] = (720, 480)) -> np.ndarray:
    """Compress image to target resolution"""
    logging.info(f"[compress_image]: Compressing to {target_resolution}")
    try:
        return cv2.resize(image, target_resolution)
    except Exception as e:
        logging.error(f"[compress_image]: Error compressing image - {e}")
        raise

def take_screenshot() -> str:
    """Take a screenshot and return it as base64 string"""

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    print(ret)
    assert ret
    
    # conver to rgb
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Add grid overlay
    print(frame.shape)
    frame_with_grid, mapping_from_cell_num_to_location = add_grid_to_image(frame)
    
    # show the image
    plt.imshow(frame_with_grid)
    plt.show()
    
    # Convert to base64
    _, buffer = cv2.imencode('.jpg', frame_with_grid)
    # Save image to a random filename in /tmp
    random_filename = f'/tmp/screenshot_{uuid.uuid4()}.jpg'
    cv2.imwrite(random_filename, cv2.cvtColor(frame_with_grid, cv2.COLOR_RGB2BGR))
    print(f"Saved screenshot to: {random_filename}")
    
    screenshot = base64.b64encode(buffer).decode('utf-8')
    
    logging.info("[take_screenshot]: Screenshot captured and processed successfully")
    return frame, mapping_from_cell_num_to_location, screenshot