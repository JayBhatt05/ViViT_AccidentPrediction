########################### Annotation Script ###########################
#Authors: Jay Ketan Bhatt, Kartikay Goel, Krish Agarwal

'''This script takes video file as input and analyses it using YOLO and DeepSort algorithms to extract
   vehicle information and output it onto a text file in a tabular format
'''


import cv2
import torch
import numpy as np
from torchvision.ops import box_iou
import os
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Load the YOLO model
model = YOLO("yolov5su.pt")

# Get the class names from the model
class_names = model.names

# Load the DeepSort tracker
tracker = DeepSort(max_age=5,
                   n_init=2,  
                   nms_max_overlap=0.7, 
                   max_cosine_distance=0.1,  
                   nn_budget=150,
                   override_track_class=None,
                   embedder="mobilenet",
                   half=True,
                   bgr=True,
                   embedder_gpu=True,
                   embedder_model_name=None,
                   embedder_wts=None,
                   polygon=False,
                   today=None)

# Input video path
video = "000950"
video_path = f"Crash_1500/{video}.mp4"
cap = cv2.VideoCapture(video_path)

# Output path for the text file
output_path = f"Crash_1500_Annotation/{video}_ann.txt"
output_dir = os.path.dirname(output_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

frame_id = 0

def calculate_iou(bbox1, bbox2):
    bbox1 = torch.tensor(bbox1).unsqueeze(0)
    bbox2 = torch.tensor(bbox2).unsqueeze(0)
    return box_iou(bbox1, bbox2).item()

with open(output_path, 'w') as output_file:
    output_file.write(f'video_no\tFrame\tVehicle_ID\tVehicle_Type\tx1\ty1\twidth\theight\tAccident\n')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        print(f"\nProcessing frame {frame_id}")

        results = model.predict(frame, conf=0.3, iou=0.2)
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf.cpu().numpy()
                cls = box.cls.cpu().numpy()
                if conf > 0.3 and cls in [0, 1, 2, 3, 7]:
                    class_name = class_names[int(cls)]
                    if class_name != "traffic light" and class_name != "person":
                        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, class_name))

        # Update tracks
        tracks = tracker.update_tracks(detections, frame=frame)

        # Store previous frame's tracks
        if frame_id > 1:
            prev_tracks = current_tracks
        current_tracks = tracks

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            width, height = x2 - x1, y2 - y1

            vehicle_type = track.get_det_class()
            if vehicle_type is None:
                vehicle_type = "unknown"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{track_id} {vehicle_type}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            output_file.write(f"{video}\t{frame_id}\t{track_id}\t{vehicle_type}\t{x1}\t{y1}\t{width}\t{height}\t0\n")

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("\nProcessing complete...\n")