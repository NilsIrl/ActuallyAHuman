import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

actions = [
    {
        "name": "click",
        "parameters": {
            "grid_number": "int"
        },
        "description": "click on the screen at the given grid number"
    },
]

def load_prompt():
    logging.info("[load_prompt]: Loading prompt from file")
    try:
        with open(__file__.replace("prompts.py", "prompt.txt"), "r") as file:
            temp_prompt = file.read()
        
        # Convert actions to a formatted string
        actions_str = get_actions()
        order = get_order()
        
        # Format the prompt with proper string escaping
        prompt = temp_prompt.format(
            actions=actions_str,
            order=order
        )
        
        logging.info("[load_prompt]: Prompt loaded and formatted successfully")
        return prompt
    except Exception as e:
        logging.error(f"[load_prompt]: Error loading prompt - {e}")
        logging.error(f"[load_prompt]: Template: {temp_prompt}")
        logging.error(f"[load_prompt]: Actions: {actions_str}")
        logging.error(f"[load_prompt]: Order: {order}")
        raise

def get_order():
    logging.info("[get_order]: Generating order")
    try:
        order = {
            "name": "burger",
            "quantity": 1
        }
        logging.info(f"[get_order]: Order created - {order}")
        return json.dumps(order)
    except Exception as e:
        logging.error(f"[get_order]: Error creating order - {e}")
        raise

def get_prompt():
    logging.info("[get_prompt]: Getting formatted prompt")
    return load_prompt()

def get_actions():
    logging.info("[get_actions]: Returning available actions")
    return json.dumps(actions, indent=2)
