import llm
from typing import Optional
import utils
import cv2
import matplotlib.pyplot as plt
import time
import asyncio
import logging
import json
import uuid
from microservices_utils import extend, initialize_arm, connect_to_arduino, pan_servo, set_arm_position
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

#### States:
# See the thank you screen -> order finished 
# See the order screen -> order in progress 
# See the menu screen -> waiting for order 

#### Actions:
# Click
    # click: {x: int, y: int} -> clicks the screen at the given coordinates of the input image 
    # each click - extend and click and then reset to og position before taking another screenshot to recalibrate state
# Scroll Up

# Scroll Down
# Swipe Left
# Swipe Right

def pass_screenshot_to_gpt(screenshot: str, conversation_id: Optional[str] = None) -> tuple[str, str]:
    """Pass screenshot to GPT and get response with conversation context"""
    logging.info("[pass_screenshot_to_gpt]: Sending screenshot to GPT")
    
    prompt = "What do you see in this screenshot? What state is the iPad in?"
    response, conv_id = llm.get_env(screenshot, prompt, conversation_id)
    
    if response:
        try:
            response = response.strip()
            logging.info(f"[pass_screenshot_to_gpt]: Raw response - {response}")
            return response, conv_id
        except Exception as e:
            logging.error(f"[pass_screenshot_to_gpt]: Error parsing response - {e}")
            return None, conv_id
            
    logging.warning("[pass_screenshot_to_gpt]: No response received")
    return None, conv_id

def get_grid_coordinates(grid_number: int) -> tuple[int, int]:
    """Get the coordinates of the grid number"""
    logging.info(f"[get_grid_coordinates]: Getting coordinates for grid number - {grid_number}")
    
    # get the grid number in the format of 10x10 grid
    row = (grid_number - 1) // 10
    col = (grid_number - 1) % 10
    
    # get the coordinates of the center of the grid
    x = 720/10 * col + 720/20
    y = 1280/10 * row + 1280/20
    
    return x, y
    

def run_order():
    conversation_id = uuid.uuid4()
    frame, mapping_from_cell_num_to_location, screenshot = utils.take_screenshot()
    assert screenshot

    response, conversation_id = pass_screenshot_to_gpt(screenshot, conversation_id)
    response = json.loads(response)

    print(response)
    grid_number = response['action']['grid_number']
    x, y, x_end, y_end = mapping_from_cell_num_to_location[grid_number]

    tracker = cv2.TrackerMIL_create()
    # ROI format for OpenCV is (x, y, width, height) as integers
    roi = (int(x), int(y), int(x_end - x), int(y_end - y))
    tracker.init(frame, roi)

    cap = cv2.VideoCapture(0)
    PX_GAIN = -0.01
    PY_GAIN = 2
    y_axis_control = initialize_arm()
    x_axis_control = connect_to_arduino()
    
    x_axis_pos = 0
    y_axis_pos = y_axis_control.ReadEncM2(0x80)
    assert y_axis_pos[0]
    y_axis_pos = y_axis_pos[1]
    
    last_update_time = time.time()  # Add timestamp tracking
    i = 0

    # while True:
    while True:
        ret, frame = cap.read()
        assert ret
        success, box = tracker.update(frame)
        # Draw the bounding box on the frame
        if success:
            # Box format is (x, y, width, height)
            p1 = (int(box[0]), int(box[1]))
            p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
            
            # Display the frame
            # Save frame to random file location
            random_filename = f'/tmp/tracking_frame_{uuid.uuid4()}.jpg'
            cv2.imwrite(random_filename, frame)
            print(f"Saved tracking frame to: {random_filename}")
        # Calculate midpoints of bounding box
        mid_x = (box[0] + box[2]) / 2
        mid_y = (box[1] + box[3]) / 2

        print(f"mid_x: {mid_x}, mid_y: {mid_y}")
        x_diff = PX_GAIN * (mid_x - 832)
        y_diff = PY_GAIN * (mid_y - 288)
        print(f"x_diff: {x_diff}, y_diff: {y_diff}")

        current_time = time.time()
        if current_time - last_update_time >= 2:  # Only update if 2 seconds elapsed
            i += 1
            x_axis_pos += int(x_diff)
            y_axis_pos += int(y_diff)

            x_axis_pos = max(-60, min(60, x_axis_pos))

            print("CANARY")
            print(f"x_axis_pos: {x_axis_pos}, y_axis_pos: {y_axis_pos}")
            set_arm_position(y_axis_control, y_axis_pos)
            pan_servo(x_axis_control, x_axis_pos)
            last_update_time = current_time  # Update the timestamp
            if i == 10:
                break

    extend(x_axis_control)

    return frame, response


def main():
    logging.info("[main]: Starting iPad automation")
    run_order()


if __name__ == "__main__":
    main()
