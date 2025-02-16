import openai
import cv2
import base64
from PIL import Image
import io
from openai import OpenAI

class GPT4VisionNavigator:
    def __init__(self, api_key):
        openai.api_key = api_key

    def encode_image_to_base64(self, image):
        """
        Convert a BGR image (numpy array) to a base64-encoded JPEG.
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        return img_b64

    def generate_navigation_instructions(self, building_image):
        """
        Given an image of a building (floorplan), prompt GPT-4 Vision mini to extract the floorplan details
        and generate step-by-step indoor navigation instructions.
        """
        img_b64 = self.encode_image_to_base64(building_image)
        prompt = (
            "You are provided with an image of a building's floorplan. "
            "Please extract the key floorplan details and generate clear, step-by-step indoor navigation "
            "instructions from the main entrance to 'Conference Room A'. "
            "Include any landmarks, corridor details, and turning instructions. "
            "Be concise and specific."
        )

        client = OpenAI(api_key="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")
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "image": img_b64,
                }
            ],
            max_tokens=300,
        )
        #return response.choices[0].message["content"]
        return response.choices[0].message.content


# Example usage:
if __name__ == "__main__":
    api_key = "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"  # Replace with your actual key.
    navigator = GPT4VisionNavigator(api_key)
    building_image = cv2.imread("floorplan1.jpg")
    if building_image is None:
        print("Error: Could not load floorplan.jpg")
    else:
        instructions = navigator.generate_navigation_instructions(building_image)
        print("Navigation Instructions:")
        print(instructions)