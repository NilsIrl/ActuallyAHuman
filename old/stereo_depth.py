import cv2
import numpy as np

# Open USB Stereo Camera (single device outputting combined frame)
cap = cv2.VideoCapture(0)  # Adjust index if needed

# StereoBM Matcher
stereo = cv2.StereoSGBM_create(
    numDisparities=16, blockSize=15,
    P1=8 * 3 * 15 ** 2, P2=32 * 3 * 15 ** 2
)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get frame dimensions
    height, width, _ = frame.shape

    # Split into left and right images (assuming side-by-side layout)
    left_img = frame[:, :width//2]
    right_img = frame[:, width//2:]

    # Convert to grayscale
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    # Compute Disparity
    disparity = stereo.compute(left_gray, right_gray)

    # Normalize for display
    disp_norm = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    # Show images
    cv2.imshow("Left Image", left_gray)
    cv2.imshow("Right Image", right_gray)
    cv2.imshow("Disparity Map", disp_norm.astype(np.uint8))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()