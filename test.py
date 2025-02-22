from ultralytics import YOLO
import cv2

model = YOLO('./runs/detect/train9/weights/best.pt')

# model('./*.mp4', show=True)
cap = cv2.VideoCapture(0)
print(cap)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, conf=0.8)
    cv2.imshow('YOLOv8 Detection', results[0].plot())
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
