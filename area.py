import cv2
import numpy as np
import torch
from ultralytics import YOLO
import pygame
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders

# Initialize YOLOv8 model with the pre-trained weights
model = YOLO('yolov8n.pt')  # Load the YOLOv8n model

# COCO pretrained dataset class names
coco_class_names = {
    0: 'person', 43: 'knife', 80: 'gun'
}

# Define your own class names
my_class_names = {
    0: 'person',
    1: 'gun',
    2: 'fight',
    3: 'violent_move',
    # Add other classes as needed
}

# Example detection function (replace with your actual detection logic)
def detect_objects(frame):
    # This function should return a list of detections
    # Each detection is a tuple (class_id, score, x1, y1, x2, y2)
    # Replace this with your actual detection code
    return [
        (0, 0.95, 50, 50, 200, 200),  # Example detection
        (1, 0.90, 300, 300, 400, 400),  # Example detection
    ]

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



cv2.namedWindow('Video')
cv2.setMouseCallback('Video', draw_polygon)

# Initialize background subtractor
kernel = np.ones((10, 10), np.uint8)
backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=True, varThreshold=50, history=2800)
thresh = 1500


person_count = 0
start_time = None

# Initialize pygame mixer
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound('alarm.mp3')

def send_email(image_path):
    fromaddr = "lucas12234567@gmail.com"
    toaddr = "lucas12234567@gmail.com"
    msg = MIMEMultipart()
    msg['From'] = fromaddr
    msg['To'] = toaddr
    msg['Subject'] = "Intruder Alert!!!"

    body = "An intruder has been detected. See the attached image."
    msg.attach(MIMEText(body, 'plain'))

    attachment = open(image_path, "rb")
    part = MIMEBase('application', 'octet-stream')
    part.set_payload((attachment).read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', "attachment; filename= %s" % image_path)
    msg.attach(part)

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(fromaddr, "nbue pgov qxkw sqwt")
    text = msg.as_string()
    server.sendmail(fromaddr, toaddr, text)
    server.quit()

# Initialize person count, timer, and email sent flag
person_count = 0
start_time = None
email_sent = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

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
                person_count = 1  # Set the person count to 1 when an intruder is detected

                # Start or update the timer
                if start_time is None:
                    start_time = time.time()
                elif time.time() - start_time >= 5 and not email_sent:
                    alarm_sound.play()
                    image_path = 'intruder.jpg'
                    cv2.imwrite(image_path, frame)
                    send_email(image_path)
                    email_sent = True  # Set the flag to indicate that the email has been sent
            else:
                person_count = 0
                start_time = None  # Reset the timer if no intruder is detected
        else:
            person_count = 0
            start_time = None  # Reset the timer if no contours are found

    # Run object detection on the frame
    detections = detect_objects(frame)

    for detection in detections:
        class_id, score, x1, y1, x2, y2 = detection
        class_name = my_class_names.get(class_id, 'Unknown')


        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f'{class_name}: {score:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if class_name in ['fight', 'violent_move']:
            cv2.putText(frame, 'Fight Detected!', (int(x1), int(y1) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the number of people detected
    cv2.putText(frame, f'People Count: {person_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow('Video', frame)

    # Add a delay to control the frame rate
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
