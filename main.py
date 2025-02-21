from ultralytics import YOLO
import cv2

model = YOLO('yolo11m.pt')

model.train(data='../Dataset/data.yaml', epochs=100, imgsz=640, device='cpu', batch=8)
