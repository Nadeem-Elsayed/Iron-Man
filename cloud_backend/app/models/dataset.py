import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

# Standard ResNet/YOLO normalizations
transform_2d = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_hagrid_dataloader(data_dir: str, batch_size: int = 32):
    """
    Creates the 2D CNN DataLoader for static poses.
    Because you structured the raw folders correctly (class_name/img.jpg), 
    we can use PyTorch's native ImageFolder for zero-friction loading.
    """
    # ImageFolder automatically assigns labels based on your folder names
    dataset = datasets.ImageFolder(root=data_dir, transform=transform_2d)
    
    # Utilizing 8 workers to saturate your Ryzen 5 CPU cores during data fetching
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    return loader, dataset.classes

class JesterSequenceDataset(Dataset):
    """
    Custom 3D Sequence DataLoader for dynamic motions.
    Filters the massive Jester CSV on the fly so you don't have to delete the raw videos.
    """
    def __init__(self, csv_path: str, frames_dir: str, target_labels: list, sequence_length: int = 15):
        self.frames_dir = frames_dir
        self.sequence_length = sequence_length
        self.transform = transform_2d
        
        # Load the CSV and filter only for the active sweeping gestures
        df = pd.read_csv(csv_path, sep=';', header=None, names=['video_id', 'label'])
        self.data = df[df['label'].isin(target_labels)].reset_index(drop=True)
        
        # Create a mapping dictionary for labels (e.g., "Swiping Left" -> 0)
        self.label_map = {label: idx for idx, label in enumerate(target_labels)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        vid_id = str(row['video_id'])
        label_idx = self.label_map[row['label']]
        vid_path = os.path.join(self.frames_dir, vid_id)
        
        # Read the sequence of .jpg frames in the video folder
        frames = []
        if os.path.exists(vid_path):
            frame_files = sorted([f for f in os.listdir(vid_path) if f.endswith('.jpg')])
            
            # Sample exactly `sequence_length` frames evenly across the video
            step = max(1, len(frame_files) // self.sequence_length)
            sampled_files = frame_files[::step][:self.sequence_length]
            
            for file_name in sampled_files:
                img_path = os.path.join(vid_path, file_name)
                img = Image.open(img_path).convert('RGB')
                frames.append(self.transform(img))
                
        # Pad with blank frames if the video was too short
        while len(frames) < self.sequence_length:
            frames.append(torch.zeros(3, 224, 224))
            
        # Stack frames into a 4D tensor: [Sequence_Length, Channels, Height, Width]
        sequence_tensor = torch.stack(frames)
        return sequence_tensor, label_idx

def get_jester_dataloader(csv_path: str, frames_dir: str, batch_size: int = 8):
    target_labels = ["Swiping Left", "Swiping Right"]
    dataset = JesterSequenceDataset(csv_path, frames_dir, target_labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return loader