import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from sklearn.preprocessing import MinMaxScaler
from torch.nn.utils.rnn import pad_sequence

# Function to load annotations
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
                cx, cy = x + w/2.0, y - h/2.0
                data.append((frame_num, vehicle_id, cx, cy))
            annotations[video_id] = data
    return annotations

# Function to normalize annotations
def normalize_annotations(annotations):
    scaler = MinMaxScaler()
    for video_id, data in annotations.items():
        coords = np.array([bbox[2:4] for bbox in data])
        scaled_coords = scaler.fit_transform(coords)
        for i, bbox in enumerate(data):
            annotations[video_id][i] = bbox[:2] + tuple(scaled_coords[i])
    return annotations

# Function to load video data
def load_video_data(video_dir, annotation_dir):
    video_data = {}
    for video_file, text_file in zip(os.listdir(video_dir), os.listdir(annotation_dir)):
        video_id = video_file[:6]
        video_path = os.path.join(video_dir, video_file)
        text_path = os.path.join(annotation_dir, text_file)
        
        # Loop to filter out videos with no accident or only 1 accident vehicle
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
        if len(ids) == 1:
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

# Collate function for batching variable-length sequences
def collate_fn(batch):
    frames_batch = [item[0] for item in batch]  # List of frame tensors
    targets_batch = [item[1] for item in batch]  # List of target tensors (vehicle centers)

    # Pad frames to the maximum number of frames in the batch
    frames_padded = pad_sequence(frames_batch, batch_first=True, padding_value=0)
    
    # Pad targets to the maximum number of vehicles in the batch
    max_vehicles = max([targets.size(0) for targets in targets_batch])
    padded_targets = []
    
    for targets in targets_batch:
        num_vehicles = targets.size(0)
        if num_vehicles < max_vehicles:
            # Pad with (0, 0) to match the maximum number of vehicles
            padded_targets.append(torch.cat([targets, torch.zeros(max_vehicles - num_vehicles, 2)], dim=0))
        else:
            padded_targets.append(targets)
    
    padded_targets = torch.stack(padded_targets)
    
    return frames_padded, padded_targets

# Custom Dataset for traffic data
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
    
    def load_targets(self, video_id, start_frame):
        annotations = [bbox for bbox in self.annotations[video_id] if bbox[0] == start_frame]
        centers = [(ann[2], ann[3]) for ann in annotations]
        return torch.tensor(centers, dtype=torch.float32)
    
    def _get_video_id_and_frame(self, index):
        cumulative_length = 0
        for video_id in self.video_data:
            video_length = len(self.video_data[video_id]) - self.window_size
            if index < cumulative_length + video_length:
                return video_id, index - cumulative_length
            cumulative_length += video_length
        raise IndexError("Index Out of Range")
    
    def __getitem__(self, index):
        video_id, start_frame = self._get_video_id_and_frame(index)
        frames = self.load_frames(video_id, start_frame)
        targets = self.load_targets(video_id, start_frame + self.window_size)
        
        return frames, targets
