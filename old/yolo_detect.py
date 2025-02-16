"""
brew install python3 cmake pkg-config
brew install opencv
python3 -m venv yolov8_env
source yolov8_env/bin/activate
pip install ultralytics torch torchvision torchaudio
pip install numpy opencv-python scipy filterpy scikit-learn
pip install opencv-contrib-python
pip install numpy opencv-python scipy filterpy scikit-learn
pip install numba --no-cache-dir  # Fix Applegit clone https://github.com/nwojke/deep_sort.git
cd deep_sort
pip install -r requirements.txt Silicon compatibility issue
pip install opencv-contrib-python


winget install Python
winget install Kitware.CMake
pip install opencv-python opencv-contrib-python
python -m venv yolov8_env
yolov8_env\Scripts\activate
pip install ultralytics torch torchvision torchaudio
pip install numpy opencv-python scipy filterpy scikit-learn
pip install opencv-contrib-python
pip install numba --no-cache-dir
git clone https://github.com/nwojke/deep_sort.git
cd deep_sort
pip install -r requirements.txt
python -c "import torch, cv2, ultralytics; print('Setup successful')"
"""

from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("YOLOv8 Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()