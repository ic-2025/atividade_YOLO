from ultralytics import YOLO
import time
import cv2

class YOLOModel:

    """
    Class to encapsulate the YOLO model and the functions to track positions of people in a video.

    initialize the YOLO model with the given model path and task.

        Args:
            model_path (str): The path to the YOLO model.
            task (str, optional): The task to perform. Defaults to 'detect'.
            CLASS_NAMES (list[str], optional): The classes to detect. Defaults to ["em_pe","deitado","sentado"].
    """
    def __init__(self, model_path: str, task: str = 'detect', CLASS_NAMES: list[str] = ["em_pe", "deitado", "sentado"]) -> None:
        """
        Initialize the YOLO model with the given model path and task.

        Args:
            model_path (str): The path to the YOLO model.
            task (str): The task to perform. Defaults to 'detect'.
            CLASS_NAMES (list of str): The classes to detect. Defaults to ["em_pe", "deitado", "sentado"].

        Returns:
            None
        """
        self.model = YOLO(model_path, task=task)
        # Define the classes to detect (COCO dataset classes)
        self.CLASS_NAMES = CLASS_NAMES  # We only care about detecting people

       

    # Function to determine the position based on bounding box height-to-width ratio
    def get_position(self, results, POSITIONS: dict[str, int]) -> None:
        """
        Gets the positions from the YOLO detection results and updates the POSITIONS dictionary.

        Args:
            results (YOLO results): The results of the YOLO detection.
            POSITIONS (dict of str to int): The dictionary to be updated with the positions.
        """
        # Get the names of the detected classes
        names = [results[0].names[cls.item()] for cls in results[0].boxes.cls.int()]
        # If no boxes were detected, increment the not_detected count
        if len(results[0].boxes) == 0:
            POSITIONS["not_detected"] += 1
            return
        # For each detected class, increment the count in POSITIONS
        for name in names:
            POSITIONS[name] += 1

    def get_track_with_yolo(
        self, 
        video_path: str, 
        CONF: float = 0.5, 
        IOU: float = 0.5
    ) -> tuple[dict[str, int], int, float]:
        """
        Runs the YOLOv11 object detection model on a video and returns the count of each position.

        Args:
            video_path (str): The path to the video file to be processed.
            CONF (float, optional): The confidence threshold for YOLOv11. Defaults to 0.5.
            IOU (float, optional): The intersection over union threshold for YOLOv11. Defaults to 0.5.

        Returns:
            tuple[dict[str, int], int, float]: A tuple containing a dictionary of positions (em_pe, sentado, deitado, not_detected) and the total frames and time taken to process the video.
        """
        # Define the positions (standing, sitting, lying down)
        POSITIONS: dict[str, int] = {
            "em_pe": 0,
            "sentado": 0,
            "deitado": 0,
            "not_detected": 0,
        }

        # Open the video capture
        cap = cv2.VideoCapture(video_path)

        # Variables for time tracking
        start_time: float = time.time()
        frame_count: int = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLOv11 object detection on the frame
            results = self.model.predict(
                frame, 
                verbose=False, 
                conf=CONF,
                iou=IOU,
                imgsz=640,
            )

            # Display the frame with bounding boxes
            cv2.imshow("Elderly Monitoring \nPressione na tecla 'q' para sair do programa.", results[0].plot())

            # Process the results
            self.get_position(results, POSITIONS)

            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

        # Release resources
        cap.release()
        cv2.destroyAllWindows()

        end_time: float = time.time()
        total_time: float = end_time - start_time
        total_frames: int = frame_count

        return (POSITIONS, total_frames, total_time)

if __name__ == '__main__':
    # Create an instance of YOLOModel
    yolo = YOLOModel('./runs/detect/train17/weights/best.pt', task='detect')

    print("#######################\n\n Bem vindo ao Elderly Monitoring\n\n#######################")

    print("\nPressione na tecla 'q' para sair do programa.\n")

    selection =  input("Deseja utilizar seu Webcam ou um arquivo de video? (s: webcam / n: video): ").lower()

    if selection == 's':
        user_input = 0 
    else:
        user_input = input("Digite o caminho do video: ")

    [POSITIONS, total_frames, total_time] = yolo.get_track_with_yolo(user_input) 

    if total_frames != 0:
    # Calculate total time and percentages
        
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
