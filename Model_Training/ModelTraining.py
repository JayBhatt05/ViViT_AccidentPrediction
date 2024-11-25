from Model import FactorizedEncoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from DataProcessing import TrafficDataset, load_annotations, normalize_annotations, load_video_data, collate_fn


def is_accident(pred, threshold=0.05):
    distances = torch.cdist(pred, pred)
    
    # Mask for handling vehicles which have either exited the frame or are yet to come
    mask = (pred[:, :, 0] != 0) & (pred[:, :, 1] != 0)
    mask = mask.unsqueeze(1) & mask.unsqueeze(2)
    
    # Create upper triangular matrix of close distances
    close_distances = torch.triu((distances < threshold) & mask, diagonal=1)
    
    # Check if there are any close distances
    return torch.any(close_distances)


def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    #best_val_loss = float("inf")
    for epoch in range(num_epochs):
        
        #Training
        model.train()
        train_loss = 0
        accidents_predicted = 0
        for frames, targets in train_loader:
            frames, targets = frames.to(device), targets.to(device)
            optimizer.zero_grad()
            pred = model(frames)
            loss = criterion(pred, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            if is_accident(pred):
                accidents_predicted += 1
                
        '''#Validation        
        model.eval()
        val_loss = 0.0
        val_accidents_predicted = 0
        with torch.no_grad():
            for frames, targets in val_loader:
                frames, targets = frames.to(device), targets.to(device)
                pred = model(frames)
                loss = criterion(pred, targets)
                val_loss += loss.item()
                
                if is_accident(pred):
                    val_accidents_predicted += 1'''
                
        train_loss /= len(train_loader)
        #val_loss /=  len(val_loader)
        
        print(f"Epoch: {epoch+1}/{num_epochs}")
        print(f"Train loss: {train_loss:.4f}, Accidents Predicted: {accidents_predicted}")
        #print(f"Val loss: {train_loss:.4f}, Accidents Predicted: {accidents_predicted}\n")
        
        #if val_loss < best_val_loss:
        #best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
            
    return model


#Hyperparameters
INPUT_WINDOW_SIZE = 5
MAX_VEHICLES = 5
EMBED_DIM = 768
NUM_HEADS = 8
SPATIAL_DEPTH = 6
TEMPORAL_DEPTH = 6
BATCH_SIZE = 8
NUM_EPOCHS = 50
LR = 0.001


#Data Loading
annotation_dir = f"Dataset/Training_Annotation"
annotations = load_annotations(annotation_dir)
annotations = normalize_annotations(annotations)
video_dir = f"Dataset/Training_Videos"
video_data = load_video_data(video_dir, annotations)


#Create Dataset and DataLoaders
dataset = TrafficDataset(video_data, annotations, INPUT_WINDOW_SIZE)
#train_size = int(0.8 * len(dataset))
#val_size = len(dataset) - train_size
#train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
#val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

#Model Specifications and Training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FactorizedEncoder(in_channels=3, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, spatial_depth=SPATIAL_DEPTH,
                         temporal_depth=TEMPORAL_DEPTH, max_vehicles=MAX_VEHICLES).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

trained_model = train_model(model, train_loader, criterion,optimizer, NUM_EPOCHS, device)