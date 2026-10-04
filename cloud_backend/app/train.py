import os
import argparse
import torch
import torch_directml
import torch.nn as nn
import torch.optim as optim
from models.dataset import get_hagrid_dataloader, get_jester_dataloader
from models.vision_2d import HaGRIDResNet
from models.vision_3d import JesterResNet3D

def train_model(model, dataloader, criterion, optimizer, num_epochs, device, save_path):
    """
    Standard PyTorch training loop. Iterates through the dataloader, 
    calculates gradients, and updates the final classification layer.
    """
    model.to(device)
    
    print(f"Starting training on {device}...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            # Move data to the hardware accelerator
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 1. Clear old gradients
            optimizer.zero_grad()
            
            # 2. Forward pass
            outputs = model(inputs)
            
            # 3. Calculate error (Loss)
            loss = criterion(outputs, labels)
            
            # 4. Backward pass (Calculate new gradients)
            loss.backward()
            
            # 5. Update weights
            optimizer.step()
            
            # Track metrics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")
                
        epoch_loss = running_loss / total_samples
        epoch_acc = correct_predictions / total_samples
        print(f"--- Epoch {epoch+1} Completed | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f} ---")
        
    # Ensure the weights directory exists before saving
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Weights securely saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="IronMan Gesture Model Trainer")
    parser.add_argument('--stream', type=str, choices=['2d', '3d'], required=True, 
                        help="Which data stream to train: '2d' for HaGRID, '3d' for Jester")
    parser.add_argument('--epochs', type=int, default=10, 
                        help="Number of full passes over the dataset")
    parser.add_argument('--lr', type=float, default=0.001, 
                        help="Learning rate for the Adam optimizer")
    args = parser.parse_args()

    # Hardware routing: Uses GPU if available, otherwise falls back to CPU
    # For AMD GPUs on Windows, PyTorch defaults to CPU unless configured via ROCm (WSL2) or DirectML
    target_gpu_index = 1 
    device = torch_directml.device(target_gpu_index)
    
    print(f"Hardware Accelerator: {torch_directml.device_name(target_gpu_index)}")
    
    # Common hyperparameters
    criterion = nn.CrossEntropyLoss()

    if args.stream == '2d':
        print("Initializing 2D Pipeline (HaGRID)...")
        data_dir = "../../data/raw/hagrid"
        dataloader, _ = get_hagrid_dataloader(data_dir=data_dir, batch_size=32)
        
        model = HaGRIDResNet(num_classes=5)
        
        # CRITICAL FIX: Move to GPU *before* creating the optimizer
        model.to(device) 
        
        optimizer = optim.Adam(model.backbone.fc.parameters(), lr=args.lr)
        save_path = "../weights/hagrid_resnet50.pth"
        
    elif args.stream == '3d':
        print("Initializing 3D Pipeline (Jester)...")
        csv_path = "../../data/raw/jester/Train.csv"
        frames_dir = "../../data/raw/jester/Train"
        
        print("1. Loading CSV and checking dataset size...")
        dataloader = get_jester_dataloader(csv_path=csv_path, frames_dir=frames_dir, batch_size=8)
        print(f" -> Found {len(dataloader.dataset)} active swipe gestures.")
        
        print("2. Instantiating ResNet3D model...")
        model = JesterResNet3D(num_classes=3)
        
        print("3. Mapping 3D model to AMD GPU... (This may take a moment)")
        model.to(device)
        print(" -> Model successfully locked to DirectML!")
        
        optimizer = optim.Adam(model.backbone.fc.parameters(), lr=args.lr)
        save_path = "../weights/jester_r3d_18.pth"

    # CRITICAL: This must be at the very bottom of main(), un-indented from the if/elif blocks!
    train_model(model, dataloader, criterion, optimizer, args.epochs, device, save_path)

if __name__ == "__main__":
    main()