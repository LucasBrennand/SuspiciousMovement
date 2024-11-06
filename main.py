import cv2
import pygame
from ultralytics import YOLO

# Initialize YOLOv8 model with the pre-trained weights
model = YOLO('yolov8n.pt')  # Load the YOLOv8n model

# COCO pretrained dataset class names
coco_class_names = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train',
    7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter',
    13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant',
    21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie',
    28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 
    34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
    39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog',
    53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
    60: 'dining table', 61: 'toilet', 62: 'TV', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator',
    73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush',
    80: 'gun'  # Add gun class if it exists in your model
}

# Ask the user if they want to open the camera or process a video file
choice = input("Gostaria de abrir a camera ou um video? (camera/video): ").strip().lower()

if choice == 'camera':
    cap = cv2.VideoCapture(0)
elif choice == 'video':
    video_path = input("Digite o arquivo do video: ").strip()
    cap = cv2.VideoCapture(video_path)
else:
    print("Escolha inválida. Saindo...")
    exit()

while True:
    ret, frame = cap.read()  # Read frame from webcam or video file
    if not ret:
        break

    # Perform object detection on the frame
    results = model.predict(source=frame)
    
    person_count = 0  # Initialize person counter

    # Loop through detections
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, score, class_id = box

            # Convert class_id to integer and get class name
            class_id = int(class_id)
            class_name = coco_class_names.get(class_id, 'Unknown')

            # If the detected object is a person, increment the counter
            if class_name == 'person':
                person_count += 1

            # Draw bounding box for every detected object
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'{class_name}: {score:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # If a knife is detected (class ID 43), trigger the buzzer
            if class_id == 0:
                cv2.putText(frame, 'Person Detected!', (int(x1), int(y1) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # If a gun is detected, print a message
            if class_name == 'gun':
                print("Gun detected!")

    # Display the number of people detected on the frame
    cv2.putText(frame, f'People Count: {person_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the frame with detections
    cv2.imshow('Security Alarm System', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()