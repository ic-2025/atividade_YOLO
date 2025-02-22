from ultralytics import YOLO
import cv2

# model = YOLO('./yolo11n.pt', task='detect')
model = YOLO('./runs/detect/train9/weights/best.pt', task='detect')

model.train(
    data='../IC.v5i.yolov11/data.yaml',
    epochs=50,
    device='0', 
    batch=16, 
    imgsz=640, 
    workers=8, 
    conf=0.6,
    iou=0.6
)
