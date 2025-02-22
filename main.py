from ultralytics import YOLO
import cv2

model = YOLO('yolo11n.pt')

model.train(data='../Dataset/data.yaml', epochs=100, imgsz=640, device='0', batch=16)
