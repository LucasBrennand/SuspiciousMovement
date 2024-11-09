import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Initialize YOLOv8 model with the pre-trained weights
model = YOLO('yolov8n.pt')  # Load the YOLOv8n model

# COCO pretrained dataset class names
coco_class_names = {
    0: 'person', 43: 'knife', 80: 'gun'
}

# Variables for polygon points
pts = []
drawing_polygon = False

choice = input("Gostaria de abrir a camera ou um video? (camera/video): ").strip().lower()

if choice == 'camera':
    cap = cv2.VideoCapture(0)
elif choice == 'video':
    video_path = input("Digite o arquivo do video: ").strip()
    cap = cv2.VideoCapture(video_path)
else:
    print("Escolha inválida. Saindo...")
    exit()

# Function to draw the polygon for the restricted area
def draw_polygon(event, x, y, flags, param):
    global pts, drawing_polygon
    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append([x, y])
        drawing_polygon = True
    elif event == cv2.EVENT_RBUTTONDOWN:  # Clear points with right-click
        pts = []
        drawing_polygon = False

# Function to check if a point is inside a polygon
def inside_polygon(point, polygon):
    result = cv2.pointPolygonTest(np.array(polygon), (point[0], point[1]), False)
    return result >= 0

# Function to preprocess the frame
def preprocess(img):
    height, width = img.shape[:2]
    ratio = height / width
    img = cv2.resize(img, (640, int(640 * ratio)))
    return img

cv2.namedWindow('Video')
cv2.setMouseCallback('Video', draw_polygon)

# Initialize background subtractor
kernel = np.ones((10, 10), np.uint8)
backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=True, varThreshold=50, history=2800)
thresh = 1500

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_detected = frame.copy()

    # Draw the polygon (restricted area) on the frame if points exist
    if len(pts) > 1:
        cv2.polylines(frame, [np.array(pts, np.int32)], isClosed=True, color=(0, 255, 255), thickness=2)

    if len(pts) > 2:  # Only apply background subtraction if polygon is defined
        # Restrict processing to the defined polygon area
        mask = np.zeros_like(frame[:, :, 0])
        cv2.fillPoly(mask, [np.array(pts, np.int32)], 255)
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

        # Apply background subtraction
        fgmask = backSub.apply(masked_frame)
        ret, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)
        fgmask = cv2.dilate(fgmask, kernel, iterations=4)

        # Detect contours within the restricted area
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            if cv2.contourArea(cnt) > thresh:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, 'Intruder Detected', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

    # Run object detection on the frame
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

            if class_name == 'gun':
                cv2.putText(frame, 'Gun Detected!', (int(x1), int(y1) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            if class_name in ['fight', 'violent_move']:
                cv2.putText(frame, 'Fight Detected!', (int(x1), int(y1) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the number of people detected
    cv2.putText(frame, f'People Count: {person_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
