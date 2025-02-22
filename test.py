from ultralytics import YOLO
import time
import cv2

model = YOLO('./runs/detect/train9/weights/best.pt', task='detect')
# model = YOLO('./yolo11n.pt')

# model.val(data='../IC.v4i.yolov11/data.yaml')


# Define the classes to detect (COCO dataset classes)
CLASS_NAMES = ["em_pe","deitado","sentado"]  # We only care about detecting people

# Define the positions (standing, sitting, lying down)
POSITIONS = {
    "em_pe": 0,
    "sentado": 0,
    "deitado": 0,
    "not_detected": 0,
}

# Function to determine the position based on bounding box height-to-width ratio
def get_position(results):
    
    names = [results[0].names[cls.item()] for cls in results[0].boxes.cls.int()]


    if "em_pe" in names:  # Standing (tall and narrow)
        POSITIONS["em_pe"] += 1
    elif "sentado" in names:  # Sitting (moderate height-to-width ratio)
        POSITIONS["sentado"] += 1
    elif "deitado" in names:  # Lying down (short and wide)
        POSITIONS["deitado"] += 1
    else:
        POSITIONS["not_detected"] += 1


# Open the video file or webcam
video_path = "6.mp4"  # Replace with your video file or 0 for webcam
cap = cv2.VideoCapture(video_path)

# Variables for time tracking
frame_count = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv11 object detection on the frame
    results = model.predict(frame, verbose=False, conf=0.5, iou=0.6)

    # Process the results
    get_position(results)
    

    # Display the frame with bounding boxes
    cv2.imshow("Elderly Monitoring", results[0].plot())

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

if frame_count != 0:
# Calculate total time and percentages
    end_time = time.time()
    total_time = end_time - start_time
    total_frames = frame_count

    # Calculate time spent in each position
    time_standing = (POSITIONS["em_pe"] / total_frames) * total_time
    time_sitting = (POSITIONS["sentado"] / total_frames) * total_time
    time_lying_down = (POSITIONS["deitado"] / total_frames) * total_time
    time_not_detected = (POSITIONS["not_detected"] / total_frames) * total_time

    sec = 60
    # Convert time from seconds to minutes
    time_standing_min = (time_standing / sec)
    time_sitting_min = (time_sitting / sec)
    time_lying_down_min = (time_lying_down / sec)
    time_not_detected_min = (time_not_detected / sec)
    total_time_min = (total_time / sec)

    # Print the results
    print(f"Tempo em pé: {time_standing_min:.2f} minutos ({time_standing / total_time * 100:.2f}%)")
    print(f"Tempo sentado: {time_sitting_min:.2f} minutos ({time_sitting / total_time * 100:.2f}%)")
    print(f"Tempo deitado: {time_lying_down_min:.2f} minutos ({time_lying_down / total_time * 100:.2f}%)")
    print(f"Tempo não detectado: {time_not_detected_min:.2f} minutos ({time_not_detected / total_time * 100:.2f}%)")
    print(f"Tempo total: {total_time/sec:.2f} minutos")

    # Release resources
    cap.release()
    cv2.destroyAllWindows()