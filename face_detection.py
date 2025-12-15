import cv2
import os
from ultralytics import YOLO
import torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model_path = 'yolov8n-face.pt'

if not os.path.exists(model_path):
    print(f"Model {model_path} not found. Downloading...")
    url = 'https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.pt'
    try:
        torch.hub.download_url_to_file(url, model_path)
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("Please manually download yolov8n-face.pt and place it in the current directory.")
        exit()

model = YOLO(model_path)

video_path = 'video_dataset/1.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error reading video {video_path}")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # conf=0.5: 置信度阈值
    # iou=0.5: NMS 阈值
    results = model.track(frame, persist=True, conf=0.5,iou=0.5)

    annotated_frame = results[0].plot()

    cv2.imshow('YOLOv8-Face Detection', annotated_frame)

    # 降低播放速度以便观察
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
