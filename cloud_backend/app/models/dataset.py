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

def get_hagrid_dataloader(data_dir: str, batch_size: int = 32, max_samples_per_class: int = None):
    """
    Creates the 2D CNN DataLoader for static poses.
    Supports balanced sampling via max_samples_per_class.
    """
    dataset = datasets.ImageFolder(root=data_dir, transform=transform_2d)
    classes = dataset.classes

    if max_samples_per_class is not None:
        import random
        from collections import defaultdict
        class_indices = defaultdict(list)
        for idx, (_, class_id) in enumerate(dataset.samples):
            class_indices[class_id].append(idx)

        selected_indices = []
        random.seed(42)
        for class_id, indices in class_indices.items():
            if len(indices) > max_samples_per_class:
                selected_indices.extend(random.sample(indices, max_samples_per_class))
            else:
                selected_indices.extend(indices)
        random.shuffle(selected_indices)
        dataset = torch.utils.data.Subset(dataset, selected_indices)
        print(f" -> Balanced HaGRID subset: {len(dataset)} images (max {max_samples_per_class} per class).")

    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=False, 
        persistent_workers=True 
    )
    return loader, classes

class JesterSequenceDataset(Dataset):
    """
    Custom 3D Sequence DataLoader for dynamic motions.
    Filters the massive Jester CSV on the fly so you don't have to delete the raw videos.
    """
    def __init__(self, csv_path: str, frames_dir: str, target_labels: list, sequence_length: int = 15, max_samples_per_class: int = None):
        self.frames_dir = frames_dir
        self.sequence_length = sequence_length
        self.transform = transform_2d
        
        # Load the CSV and filter only for the active sweeping gestures
        df = pd.read_csv(csv_path)
        df_filtered = df[df['label'].isin(target_labels)].reset_index(drop=True)

        # Critical Fix: Filter out rows whose video folders were not extracted to disk
        df_filtered = df_filtered[df_filtered['video_id'].apply(
            lambda vid: os.path.isdir(os.path.join(frames_dir, str(vid)))
        )].reset_index(drop=True)
        if max_samples_per_class is not None:
            balanced = []
            for label in target_labels:
                sub = df_filtered[df_filtered['label'] == label]
                if len(sub) > max_samples_per_class:
                    sub = sub.sample(n=max_samples_per_class, random_state=42)
                balanced.append(sub)
            self.data = pd.concat(balanced).sample(frac=1.0, random_state=42).reset_index(drop=True)
            print(f" -> Balanced Jester subset: {len(self.data)} clips (max {max_samples_per_class} per class).")
        else:
            self.data = df_filtered
        
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
                with Image.open(img_path) as img:
                    frames.append(self.transform(img.convert('RGB')))
                
        # Pad with blank frames if the video was too short
        while len(frames) < self.sequence_length:
            frames.append(torch.zeros(3, 224, 224))
            
        # Stack frames into a 4D tensor: [Sequence_Length, Channels, Height, Width]
        sequence_tensor = torch.stack(frames)
        
        # CRITICAL FIX: Reorder dimensions to [Channels, Sequence_Length, Height, Width]
        sequence_tensor = sequence_tensor.permute(1, 0, 2, 3)
        
        return sequence_tensor, label_idx

def get_jester_dataloader(csv_path: str, frames_dir: str, batch_size: int = 4, max_samples_per_class: int = None):
    # The 4 explicit classes for the 3D pipeline
    target_labels = [
        "Swiping Left",       # Index 0
        "Swiping Right",      # Index 1
        "No gesture",         # Index 2 (Control - Still)
        "Doing other things"  # Index 3 (Control - Moving)
    ]
    dataset = JesterSequenceDataset(csv_path, frames_dir, target_labels, max_samples_per_class=max_samples_per_class)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,   # BUG FIX: was hardcoded to 8, ignoring the parameter.
                                 # Default lowered to 4: effective ResNet batch = 4×15 = 60
                                 # (vs. 120 at batch_size=8 which caused VRAM crash).
        shuffle=True,
        num_workers=4,
        pin_memory=False,        # CRITICAL: keep False to prevent Windows BSOD
        persistent_workers=True
    )
    return loader