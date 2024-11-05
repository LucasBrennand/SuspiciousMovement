from ultralytics import YOLO

#Load a pretrained YOLOV8 model
model = YOLO('yolov8n.pt')

#Run the external camera source
results = model(source=1, show=True, save=True, conf=0.4)





























#--------------------------------------------------------------------------------------------->


# import cv2

# from ultralytics import YOLO, solutions

# model = YOLO("yolov8n.pt")
# cap = cv2.VideoCapture(r'D:\college exams notes\final year project\YOLO_V8Project\ultralytics\fightFootage.mp4')
# assert cap.isOpened(), "Error reading video file"
# w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# # Define region points
# region_points = [(0, 400), (1080, 400), (1080, 360), (0, 360)]

# # Video writer
# video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# # Init Object Counter
# counter = solutions.ObjectCounter(
#     view_img=True,
#     reg_pts=region_points,
#     classes_names=model.names,
#     draw_tracks=True,
#     line_thickness=2,
# )

# while cap.isOpened():
#     success, im0 = cap.read()
#     if not success:
#         print("Video frame is empty or video processing has been successfully completed.")
#         break
#     tracks = model.track(im0, persist=True, show=False)

#     im0 = counter.start_counting(im0, tracks)
#     video_writer.write(im0)

# cap.release()
# video_writer.release()
# cv2.destroyAllWindows()



#--------------------------------------------------------------------------------------------->


# import cv2

# from ultralytics import YOLO, solutions
# from ultralytics.solutions import object_counter

# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# model = YOLO("yolov8m.pt")

# cap = cv2.VideoCapture(r'D:\college exams notes\final year project\YOLO_V8Project\ultralytics\fightFootage.mp4')

# counter = object_counter.ObjectCounter() # Init Object Counter
# region_points = [(0, 400), (1080, 400), (1080, 360), (0, 360)]
# counter.set_args(view_img=True, reg_pts=region_points,
#                  classes_names=model.names, draw_tracks=True)
# while cap.isOpened():
#     success, frame = cap.read()
#     if not success:
#         exit(0)
#     tracks = model.track(frame, persist=True, show=False)
#     counter.start_counting(frame, tracks)


#--------------------------------------------------------------------------------------------->


# import cv2
# from ultralytics import YOLO
# from ultralytics.solutions import object_counter # Assuming this is the module containing ObjectCounter
# cap = cv2.VideoCapture(r'D:\college exams notes\final year project\YOLO_V8Project\ultralytics\fightFootage.mp4')
# # Load a model
# model = YOLO('yolov8n.pt')  # Replace with your YOLOv8 model

# # Provide the path to the video file using a raw string or forward slashes
# videopath = r'D:\college exams notes\final year project\YOLO_V8Project\ultralytics\footageCrowd.mp4'


# # Perform tracking
# results = model.track(source=videopath, show=True, tracker="bytetrack.yaml")

# # Class names (example, replace with actual classes used by your model)
# classesNames = ['person', 'knofe', 'gun']

# # Initialize Object Counter with classes_names
# counter = object_counter.ObjectCounter(classes_names=classesNames)  # Init Object Counter
# while cap.isOpened():
#     success, frame = cap.read()
#     if not success:
#         exit(0)
#     tracks = model.track(frame, persist=True, show=False)
#     counter.start_counting(frame, tracks)



#--------------------------------------------------------------------------------------------->


# import cv2
# from ultralytics import YOLO
# from ultralytics.solutions import object_counter  # Assuming this is the module containing ObjectCounter

# # Provide the path to the video file using a raw string or forward slashes
# videopath = r'D:\college exams notes\final year project\YOLO_V8Project\ultralytics\footageCrowd.mp4'

# # Initialize video capture
# cap = cv2.VideoCapture(videopath)
# if not cap.isOpened():
#     print("Error: Could not open video.")
#     exit()

# # Load a model
# model = YOLO('yolov8n.pt')  # Replace with your YOLOv8 model

# # Perform tracking on the entire video (for visualization purposes)
# results = model.track(source=videopath, show=True, tracker="bytetrack.yaml")

# # Class names (example, replace with actual classes used by your model)
# classesNames = ['person', 'knife', 'gun']

# # Initialize Object Counter with classes_names
# counter = object_counter.ObjectCounter(classes_names=classesNames)  # Init Object Counter

# while cap.isOpened():
#     success, frame = cap.read()
#     if not success:
#         break  # Exit the loop if there are no more frames

#     # Perform prediction on the current frame
#     tracks = model.predict(frame)

#     # Perform object counting on the current frame
#     counter.start_counting(frame, tracks)

#     # Display the frame with the tracking results
#     cv2.imshow('Frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture and close all OpenCV windows
# cap.release()
# cv2.destroyAllWindows()




#--------------------------------------------------------------------------------------------->

## Specific classes

# import cv2

# from ultralytics import YOLO, solutions

# model = YOLO("yolov8n.pt")
# # cap = cv2.VideoCapture(r'D:\college exams notes\final year project\YOLO_V8Project\ultralytics\runfootage.mp4')
# cap = cv2.VideoCapture(0)
# assert cap.isOpened(), "Error reading video file"
# w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# line_points = [(20, 400), (1080, 400)]  # line or region points
# classes_to_count = [0, 2]  # person and car classes for count

# # Video writer
# video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# # Init Object Counter
# counter = solutions.ObjectCounter(
#     view_img=True,
#     reg_pts=line_points,
#     classes_names=model.names,
#     draw_tracks=True,
#     line_thickness=2,
# )

# while cap.isOpened():
#     success, im0 = cap.read()
#     if not success:
#         print("Video frame is empty or video processing has been successfully completed.")
#         break
#     tracks = model.track(im0, persist=True, show=False, classes=classes_to_count)

#     im0 = counter.start_counting(im0, tracks)
#     video_writer.write(im0)

# cap.release()
# video_writer.release()
# cv2.destroyAllWindows()


#------------------------------------------------------------------>

# Object cropping

#------------------------------------------------------------------>


# import os

# import cv2

# from ultralytics import YOLO
# from ultralytics.utils.plotting import Annotator, colors

# model = YOLO("yolov8n.pt")
# names = model.names

# # cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(r'D:\college exams notes\final year project\YOLO_V8Project\ultralytics\footageCrowd.mp4')

# assert cap.isOpened(), "Error reading video file"
# w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# crop_dir_name = "ultralytics_crop"
# if not os.path.exists(crop_dir_name):
#     os.mkdir(crop_dir_name)

# # Video writer
# video_writer = cv2.VideoWriter("object_cropping_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# idx = 0
# while cap.isOpened():
#     success, im0 = cap.read()
#     if not success:
#         print("Video frame is empty or video processing has been successfully completed.")
#         break

#     results = model.predict(im0, show=False)
#     boxes = results[0].boxes.xyxy.cpu().tolist()
#     clss = results[0].boxes.cls.cpu().tolist()
#     annotator = Annotator(im0, line_width=2, example=names)

#     if boxes is not None:
#         for box, cls in zip(boxes, clss):
#             idx += 1
#             annotator.box_label(box, color=colors(int(cls), True), label=names[int(cls)])

#             crop_obj = im0[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])]

#             cv2.imwrite(os.path.join(crop_dir_name, str(idx) + ".png"), crop_obj)

#     cv2.imshow("ultralytics", im0)
#     video_writer.write(im0)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# video_writer.release()
# cv2.destroyAllWindows()





# # Install necessary libraries
# # pip install ultralytics opencv-python pygame

# import cv2
# import pygame
# from ultralytics import YOLO

# # Initialize YOLOv8 model with the pre-trained weights
# model = YOLO('yolov8n.pt')  # Load the YOLOv8n model (you can use other variants like yolov8s.pt)

# # COCO class names
# coco_class_names = {
#     0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train',
#     7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter',
#     13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant',
#     21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie',
#     28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 
#     34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
#     39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
#     46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog',
#     53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
#     60: 'dining table', 61: 'toilet', 62: 'TV', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
#     67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator',
#     73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
# }

# # Initialize PyGame for sound
# pygame.mixer.init()
# buzzer_sound = pygame.mixer.Sound('buzzer.wav')  # Load your buzzer sound file

# # Function to play buzzer
# def play_buzzer():
#     pygame.mixer.Sound.play(buzzer_sound)

# # Start capturing video from webcam
# cap = cv2.VideoCapture(1)

# while True:
#     ret, frame = cap.read()  # Read frame from webcam
#     if not ret:
#         break

#     # Perform object detection on the frame
#     results = model.predict(source=frame)

#     # Loop through detections
#     for result in results:
#         for box in result.boxes.data:
#             x1, y1, x2, y2, score, class_id = box

#             # Convert class_id to integer and get class name
#             class_id = int(class_id)
#             class_name = coco_class_names.get(class_id, 'Unknown')

#             # Draw bounding box for every detected object
#             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#             cv2.putText(frame, f'{class_name}: {score:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#             # If a knife is detected (class ID 43), trigger the buzzer
#             if class_id == 43:
#                 play_buzzer()
#                 cv2.putText(frame, 'Knife Detected!', (int(x1), int(y1) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

#     # Display the frame with detections
#     cv2.imshow('Security Alarm System', frame)

#     # Exit if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()
# pygame.mixer.quit()




# # Install necessary libraries
# # pip install ultralytics opencv-python pygame

# import cv2
# import pygame
# from ultralytics import YOLO

# # Initialize YOLOv8 model with the pre-trained weights
# model = YOLO('yolov8n.pt')  # Load the YOLOv8n model (you can use other variants like yolov8s.pt)

# # COCO class names
# coco_class_names = {
#     0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train',
#     7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter',
#     13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant',
#     21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie',
#     28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 
#     34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
#     39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
#     46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog',
#     53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
#     60: 'dining table', 61: 'toilet', 62: 'TV', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
#     67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator',
#     73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
# }

# # Initialize PyGame for sound
# pygame.mixer.init()
# buzzer_sound = pygame.mixer.Sound('buzzer.wav')  # Load your buzzer sound file

# # Function to play buzzer
# def play_buzzer():
#     pygame.mixer.Sound.play(buzzer_sound)

# # Start capturing video from a file
# cap = cv2.VideoCapture('footageCrowd.mp4')  # Replace with your video file path
# cap = cv2.VideoCapture(1)  # Replace with your video file path

# while True:
#     ret, frame = cap.read()  # Read frame from video file
#     if not ret:
#         break

#     # Perform object detection on the frame
#     results = model.predict(source=frame)
    
#     person_count = 0  # Initialize person counter

#     # Loop through detections
#     for result in results:
#         for box in result.boxes.data:
#             x1, y1, x2, y2, score, class_id = box

#             # Convert class_id to integer and get class name
#             class_id = int(class_id)
#             class_name = coco_class_names.get(class_id, 'Unknown')

#             # If the detected object is a person, increment the counter
#             if class_name == 'person':
#                 person_count += 1

#             # Draw bounding box for every detected object
#             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#             cv2.putText(frame, f'{class_name}: {score:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#             # If a knife is detected (class ID 43), trigger the buzzer
#             if class_id == 43:
#                 play_buzzer()
#                 cv2.putText(frame, 'Knife Detected!', (int(x1), int(y1) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

#     # Display the number of people detected on the frame
#     cv2.putText(frame, f'People Count: {person_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

#     # Display the frame with detections
#     cv2.imshow('Security Alarm System', frame)

#     # Exit if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()
# pygame.mixer.quit()
