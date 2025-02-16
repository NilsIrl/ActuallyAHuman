import cv2
import os

# === USER SETTINGS ===
SAVE_DIR = "stereo_images"  # Folder to save images
CAMERA_INDEX = 0            # Change this if your camera is on a different index
FRAME_WIDTH = 1280          # Adjust based on your camera's output
FRAME_HEIGHT = 480          # Adjust based on your camera's output
KEY_CAPTURE = 'c'           # Press 'c' to capture
KEY_QUIT = 'q'              # Press 'q' to quit

# Ensure save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

# Open the camera
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    print("[ERROR] Could not open stereo camera.")
    exit()

# Function to generate the next available filename
def get_next_filename():
    index = 1
    while True:
        left_filename = os.path.join(SAVE_DIR, f"left_{index:02d}.jpg")
        right_filename = os.path.join(SAVE_DIR, f"right_{index:02d}.jpg")
        if not os.path.exists(left_filename) and not os.path.exists(right_filename):
            return left_filename, right_filename
        index += 1

print("[INFO] Press 'c' to capture an image pair. Press 'q' to quit.")

while True:
    # Capture frame from camera
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to capture image.")
        break

    # Split into left and right images
    h, w, _ = frame.shape
    half_w = w // 2
    left_img = frame[:, :half_w]   # Left half
    right_img = frame[:, half_w:]  # Right half

    # Show both images side by side
    cv2.imshow("Left Image", left_img)
    cv2.imshow("Right Image", right_img)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF

    if key == ord(KEY_CAPTURE):
        # Get next filenames
        left_filename, right_filename = get_next_filename()
        
        # Save images
        cv2.imwrite(left_filename, left_img)
        cv2.imwrite(right_filename, right_img)

        print(f"[SAVED] {left_filename}, {right_filename}")

    elif key == ord(KEY_QUIT):
        print("[INFO] Exiting...")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()