from openai import OpenAI
import uuid
from typing import Dict, List, Union
import copy
import time
import os
from prompts import get_prompt
API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=API_KEY)

# Load the system prompt
base_model = "gpt-4o-mini"
temperature = 0
max_tokens = 128

# Dictionary to store conversation histories
conversation_threads: Dict[str, List[dict]] = {}

# Time in seconds after which a conversation is considered expired
CONVERSATION_EXPIRY = 3600  # 1 hour



def create_conversation():
    """Create a new conversation thread and return its ID"""
    
    conversation_id = str(uuid.uuid4())
    system_prompt = get_prompt()
    conversation_threads[conversation_id] = {
        'messages': [{
            "role": "system",
            "content": system_prompt
        }],
        'last_updated': time.time()
    }
    return conversation_id

def get_conversation(conversation_id: str):
    """Get a conversation thread by ID, create new if not found or expired"""
    if conversation_id not in conversation_threads:
        return create_conversation()
    
    # Check if conversation has expired
    if time.time() - conversation_threads[conversation_id]['last_updated'] > CONVERSATION_EXPIRY:
        return create_conversation()
    
    return conversation_id

def add_message_to_conversation(conversation_id: str, role: str, content: Union[str, list]):
    """Add a message to a conversation thread"""
    if conversation_id not in conversation_threads:
        conversation_id = create_conversation()
    
    conversation_threads[conversation_id]['messages'].append({
        "role": role,
        "content": content
    })
    conversation_threads[conversation_id]['last_updated'] = time.time()
    return conversation_id

def get_env(image_base64: str, prompt: str, conversation_id: str = None) -> tuple[str, str]:
    """
    Function to send the prompt along with a Base64 encoded image to OpenAI API.
    Returns the response and conversation ID.
    """
    # Get or create conversation thread
    conversation_id = get_conversation(conversation_id)
    
    # Structure the message with image
    message_content = [
        {
            "type": "text",
            "text": prompt
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}"
            }
        }
    ]
    
    # Add message to conversation history
    add_message_to_conversation(conversation_id, "user", message_content)
    
    try:
        # Get full conversation history
        messages = conversation_threads[conversation_id]['messages']
        
        # Use the regular client
        completion = client.chat.completions.create(
            model=base_model,
            messages=messages,
            temperature=0.9,
        )
        
        # Add assistant's response to conversation history
        response = completion.choices[0].message.content
        add_message_to_conversation(conversation_id, "assistant", response)
        
        return response, conversation_id
        
    except Exception as e:
        print(f"Error in OpenAI call: {e}")
        return None, conversation_id

def cleanup_expired_conversations():
    """Remove expired conversation threads"""
    current_time = time.time()
    expired_ids = [
        conv_id for conv_id, conv_data in conversation_threads.items()
        if current_time - conv_data['last_updated'] > CONVERSATION_EXPIRY
    ]
    
    for conv_id in expired_ids:
        del conversation_threads[conv_id]

# Run cleanup periodically (you might want to run this in a separate thread)
def periodic_cleanup():
    while True:
        cleanup_expired_conversations()
        time.sleep(3600)  # Clean up every hour

# Optional: Start cleanup in a separate thread
# import threading
# cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
# cleanup_thread.start()

