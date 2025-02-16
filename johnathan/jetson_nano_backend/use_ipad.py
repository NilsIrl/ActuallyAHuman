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

def take_screenshot() -> Optional[str]:
    """Take a screenshot and return it as base64 string"""
    logging.info("[take_screenshot]: Taking screenshot")
    try:
        screenshot = utils.take_screenshot()
        if screenshot:
            logging.info("[take_screenshot]: Screenshot captured successfully")
        return screenshot
    except Exception as e:
        logging.error(f"[take_screenshot]: Error - {e}")
        return None

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

def perform_action(action: str) -> bool:
    """Perform the specified action on the iPad"""
    
    logging.info(f"[perform_action]: Attempting action - {action}")
    
    if action["action"] == "click":
        # draw a circle on the image at the coordinates
        screenshot = take_screenshot()
        if screenshot:
            screenshot = utils.add_grid_to_image(screenshot)
            plt.imshow(screenshot)
            plt.scatter(action["x"], action["y"], color="red", marker="o")
            plt.show()
    elif action["action"] == "scroll":
        if action["direction"] == "up":
            print("Scrolling up")
            #utils.scroll_up()
            
        elif action["direction"] == "down":
            print("Scrolling down")
            #utils.scroll_down()
    
    try:
        print(f"Performing action: {action}")
        logging.info(f"[perform_action]: Action completed - {action}")
        return True
    except Exception as e:
        logging.error(f"[perform_action]: Error performing action - {e}")
        return False
    
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
    try:
        screenshot = take_screenshot()
        if screenshot:
            response, conversation_id = pass_screenshot_to_gpt(screenshot, conversation_id)
            if response:
                logging.info(f"[main]: Received response - {response}")
                # attempt to parse as json
                try:    
                    response = json.loads(response)
                    print(response)
                except json.JSONDecodeError:
                    logging.error(f"[main]: Error parsing response as JSON - {response}")
                    
                if response["action"] == "complete":
                    logging.info("[main]: Order complete")
                    return True
                action = response["action"]
                perform_action(action)
                
    except Exception as e:
        logging.error(f"[main]: Error in main loop - {e}")
        return False
    



def main():
    logging.info("[main]: Starting iPad automation")
    Flag = True
    while Flag:
        Flag = run_order()
        
        time.sleep(1)

if __name__ == "__main__":
    main()



