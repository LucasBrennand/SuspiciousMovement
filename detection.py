import cv2
import pygame
import numpy as np
import torch
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
    80: 'gun'
}
count = 0
number_of_photos = 3
pts = []

choice = input("Gostaria de abrir a camera ou um video? (camera/video): ").strip().lower()

if choice == 'camera':
    cap = cv2.VideoCapture(0)
elif choice == 'video':
    video_path = input("Digite o arquivo do video: ").strip()
    cap = cv2.VideoCapture(video_path)
else:
    print("Escolha inválida. Saindo...")
    exit()

def draw_polygon(event, x, y, flags, param):
    global pts
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,y)
        pts.append([x, y])
    elif event == cv2.EVENT_RBUTTONDOWN:
        pts = []
    
# Function to check if a point is inside a polygon
def inside_polygon(point,polygon):
    result = cv2.pointPolygonTest(polygon, (point[0], point[1]), False)
    if result == 1:
        return True
    else:
        return False
    
def preprocess(img):
    height, width = img.shape[:2]
    ratio = height / width
    img = cv2.resize(img, (640, int(640 * ratio)))

    return img

cv2.namedWindow('Video')
cv2.setMouseCallback('Video', draw_polygon)


while True:
    ret, frame = cap.read()
    frame_detected = frame.copy()
    if not ret:
        break

    results = model.predict(source=frame)
    person_count = 0

    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, score, class_id = box

            class_id = int(class_id)
            class_name = coco_class_names.get(class_id, 'Unknown')

            if class_name == 'person':
                person_count += 1

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'{class_name}: {score:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if class_id == 0:
                cv2.putText(frame, 'Person Detected!', (int(x1), int(y1) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # If a gun is detected, print a message
            if class_name == 'gun':
                cv2.putText(frame, 'Gun Detected!', (int(x1), int(y1) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            if class_name in ['fight', 'violent_move']:
                cv2.putText(frame, 'Fight Detected!', (int(x1), int(y1) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                
        center_x = None
        center_y = None

    # Display the number of people detected on the frame
    cv2.putText(frame, f'People Count: {person_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


    # Display the frame with detections
    cv2.imshow('Security Alarm System', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()