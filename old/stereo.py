import cv2
import numpy as np
import glob
import os

# === USER SETTINGS ===
CHESSBOARD_SIZE = (9, 6)  # Number of inner corners (cols, rows)
SQUARE_SIZE = 25.0        # Square size in mm (or consistent unit)
IMAGE_DIR = "stereo_images"  # Folder where images are stored
DISPLAY_CORNERS = True     # Set to True to visualize corners during calibration

# === Prepare Calibration Data ===
termination_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare known 3D points for the chessboard
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# Store 3D and 2D points from all image pairs
objpoints = []  # 3D points in real world
imgpoints_l = []  # 2D points for left camera
imgpoints_r = []  # 2D points for right camera

# Gather image file paths
left_images = sorted(glob.glob(os.path.join(IMAGE_DIR, "left_*.jpg")))
right_images = sorted(glob.glob(os.path.join(IMAGE_DIR, "right_*.jpg")))

if len(left_images) != len(right_images) or len(left_images) == 0:
    print("[ERROR] No valid image pairs found! Ensure you have left and right images.")
    exit()

print(f"[INFO] Found {len(left_images)} stereo image pairs.")

# === Process Each Image Pair ===
for left_img_path, right_img_path in zip(left_images, right_images):
    img_l = cv2.imread(left_img_path)
    img_r = cv2.imread(right_img_path)

    if img_l is None or img_r is None:
        print(f"[WARNING] Could not read: {left_img_path}, {right_img_path}")
        continue

    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret_l, corners_l = cv2.findChessboardCorners(gray_l, CHESSBOARD_SIZE, None)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, CHESSBOARD_SIZE, None)

    if ret_l and ret_r:
        # Refine corner locations for better accuracy
        corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), termination_criteria)
        corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), termination_criteria)

        objpoints.append(objp)
        imgpoints_l.append(corners_l)
        imgpoints_r.append(corners_r)

        if DISPLAY_CORNERS:
            cv2.drawChessboardCorners(img_l, CHESSBOARD_SIZE, corners_l, ret_l)
            cv2.drawChessboardCorners(img_r, CHESSBOARD_SIZE, corners_r, ret_r)
            cv2.imshow('Left Chessboard', img_l)
            cv2.imshow('Right Chessboard', img_r)
            cv2.waitKey(500)
    else:
        print(f"[WARNING] Chessboard not found in: {left_img_path}, {right_img_path}")

cv2.destroyAllWindows()

# === Calibrate Individual Cameras ===
print("[INFO] Calibrating left camera...")
ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(objpoints, imgpoints_l, gray_l.shape[::-1], None, None)

print("[INFO] Calibrating right camera...")
ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(objpoints, imgpoints_r, gray_r.shape[::-1], None, None)

# === Stereo Calibration ===
print("[INFO] Running stereo calibration...")
flags = cv2.CALIB_FIX_INTRINSIC
criteria_stereo = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_l, imgpoints_r,
    mtx_l, dist_l, mtx_r, dist_r,
    gray_l.shape[::-1],
    criteria=criteria_stereo,
    flags=flags
)

# === Rectification & Undistortion Maps ===
print("[INFO] Computing rectification transforms...")
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    cameraMatrix1, distCoeffs1,
    cameraMatrix2, distCoeffs2,
    gray_l.shape[::-1],
    R, T,
    alpha=0  # 0 for zoomed rectification, 1 for full
)

# Compute remap matrices
left_map1, left_map2 = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, gray_l.shape[::-1], cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, gray_r.shape[::-1], cv2.CV_16SC2)

# === Save Calibration Results ===
calibration_data = {
    'cameraMatrix1': cameraMatrix1,
    'distCoeffs1': distCoeffs1,
    'cameraMatrix2': cameraMatrix2,
    'distCoeffs2': distCoeffs2,
    'R': R, 
    'T': T,
    'E': E,
    'F': F,
    'R1': R1,
    'R2': R2,
    'P1': P1,
    'P2': P2,
    'Q': Q,
    'roi1': roi1,
    'roi2': roi2,
    'left_map1': left_map1,
    'left_map2': left_map2,
    'right_map1': right_map1,
    'right_map2': right_map2
}

output_file = "stereo_calibration.npz"
np.savez(output_file, **calibration_data)
print(f"[INFO] Saved calibration to {output_file}")

# === Test Rectification ===
print("[INFO] Testing rectification...")
test_left = cv2.imread(left_images[0])
test_right = cv2.imread(right_images[0])

undistorted_left = cv2.remap(test_left, left_map1, left_map2, cv2.INTER_LINEAR)
undistorted_right = cv2.remap(test_right, right_map1, right_map2, cv2.INTER_LINEAR)

# Show rectified images
cv2.imshow("Rectified Left", undistorted_left)
cv2.imshow("Rectified Right", undistorted_right)
cv2.waitKey(0)
cv2.destroyAllWindows()