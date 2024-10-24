import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
import cv2

def load_annotations(annotation_dir):
    annotations = {}
    for file_name in os.listdir(annotation_dir):
        video_id = file_name[:6]
        with open(os.path.join(annotation_dir, file_name)) as f:
            data = []
            flag = 0
            for line in f:
                _, frame_num, vehicle_id, _, x, y, w, h, _ = line.split()
                if flag == 0:
                    flag = 1
                    continue
                frame_num = int(frame_num)
                vehicle_id = int(vehicle_id)
                x = int(x)
                y = int(y)
                w = int(w)
                h = int(h)
                #frame_num, vehicle_id, x1, y1, x2, y2 = map(int, line.split())
                cx, cy = x + w/2.0, y + h/2.0
                #cx, cy = (x1 + x2)/2.0, (y1 + y2)/2.0
                data.append((frame_num, vehicle_id, cx, cy))
            annotations[video_id] = data
            
    return annotations


def normalize_annotations(annotations):
    scaler = MinMaxScaler()
    for video_id, data in annotations.items():
        coords = np.array([bbox[2:4] for bbox in data])
        scaled_coords = scaler.fit_transform(coords)
        for i, bbox in enumerate(data):
            annotations[video_id][i] = bbox[:2] + tuple(scaled_coords[i])
            
    return annotations


def load_video_data(video_dir, annotation_dir):
    video_data = {}
    for video_file, text_file in zip(os.listdir(video_dir), os.listdir(annotation_dir)):
        video_id = video_file[:6]
        video_path = os.path.join(video_dir, video_file)
        text_path = os.path.join(annotation_dir, text_file)
        
        #Loop to filter out videos with no accident or only 1 accident vehicle
        with open(text_path) as f:
            ids = set()
            flag = 0
            for line in f:
                if flag == 0:
                    flag = 1
                    continue
                _, frame_num, vehicle_id, _, x, y, w, h, label = line.split()
                label = int(label)
                if label == 1:
                    ids.add(vehicle_id)    
        if len(ids) <= 1:
            continue    
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            frame = cv2.resize(frame, (224, 224))  # Resize to match model input
            frame = torch.tensor(frame).permute(2, 0, 1)  # Convert to (C, H, W) format
            frames.append(frame)
        cap.release()
        video_data[video_id] = frames
        
    return video_data        

    
class TrafficDataset(Dataset):
    def __init__(self, video_data, annotations, window_size):
        self.video_data = video_data
        self.annotations = annotations
        self.window_size = window_size
        
    def __len__(self):
        total_length = 0
        for video_id in self.video_data:
            total_length += (len(self.video_data[video_id]) - self.window_size)
        return total_length
    
    def load_frames(self, video_id, start_frame):
        return torch.stack(self.video_data[video_id][start_frame:start_frame + self.window_size])
    
    def load_targets(self, video_id, target_frame):
        annotations = self.annotations[video_id][target_frame]
        centers = [(ann[2], ann[3]) for ann in annotations]
        
        return torch.tensor(centers, dtype=torch.float32)
    
    def _get_video_id_and_frame(self, index):
        cumulative_length = 0
        for video_id in self.video_data:
            video_length = len(self.video_data[video_id]) - self.window_size
            if index < cumulative_length + video_length:
                return video_id, index - cumulative_length
            cumulative_length += video_id
        raise IndexError("Index Out of Range")
    
    def __getitem__(self, index):
        video_id, start_frame = self._get_video_id_and_frame(index)
        frames = self.load_frames(video_id, start_frame)
        targets = self.load_targets(video_id, start_frame + self.window_size)
        
        return frames, targets