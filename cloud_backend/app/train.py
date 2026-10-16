#!/usr/bin/env python
import os
import gc
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix

from models.dataset import get_hagrid_dataloader, get_jester_dataloader
from models.vision_2d import HaGRIDResNet
from models.vision_3d import JesterSequenceLSTM


def get_device():
    """
    Modular hardware accelerator selection:
    1. Checks for DirectML on Windows (locking onto discrete AMD Radeon RX 7800 XT).
    2. Falls back to CUDA if running on Linux/AWS GPU instance.
    3. Falls back to CPU if running in a CPU Docker container.
    """
    try:
        import torch_directml
        num_dml_devices = torch_directml.device_count()
        if num_dml_devices > 0:
            print(f"Found {num_dml_devices} DirectML devices:")
            target_device_idx = 0
            for i in range(num_dml_devices):
                name = torch_directml.device_name(i)
                print(f"  Device {i}: {name}")
                if "7800" in name or "XT" in name:
                    target_device_idx = i
            device = torch_directml.device(target_device_idx)
            print(f"--> Locked Hardware Accelerator: {torch_directml.device_name(target_device_idx)}")
            return device
    except Exception:
        pass

    if torch.cuda.is_available():
        print("--> Locked Hardware Accelerator: CUDA (NVIDIA GPU)")
        return torch.device("cuda")

    print("--> Hardware Accelerator: CPU")
    return torch.device("cpu")


def resolve_path(rel_path: str) -> str:
    """
    Resolves paths safely whether train.py is launched from the workspace root
    or from inside the cloud_backend/app directory.
    """
    if os.path.exists(rel_path):
        return os.path.abspath(rel_path)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.abspath(os.path.join(script_dir, rel_path))
    if os.path.exists(candidate) or os.path.exists(os.path.dirname(candidate)):
        return candidate
    return os.path.abspath(rel_path)


def train_model(model, dataloader, criterion, optimizer, num_epochs, device, save_path, scheduler=None, target_names=None):
    """
    PyTorch training loop with per-epoch classification reports, confusion matrix,
    per-epoch checkpoint saving, and garbage collection for rock-solid stability.
    """
    model.to(device)
    print(f"\nStarting training for {num_epochs} epochs on {device}...")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_samples = 0

        # Tracking lists for the classification report
        all_preds = []
        all_labels = []

        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            if inputs.dim() == 5 and inputs.size(2) == 3:
                inputs = inputs.permute(0, 2, 1, 3, 4)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            total_samples += labels.size(0)

            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if batch_idx % 20 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

        # Epoch Metrics
        epoch_loss = running_loss / total_samples
        print(f"\n--- Epoch {epoch+1}/{num_epochs} Completed | Loss: {epoch_loss:.4f} ---")

        report_target_names = target_names or ["Class 0", "Class 1", "Class 2", "Class 3"]
        print(classification_report(
            all_labels,
            all_preds,
            target_names=report_target_names,
            zero_division=0
        ))
        print("Confusion Matrix:\n", confusion_matrix(all_labels, all_preds))
        print("-" * 50)

        # Checkpoint: Save at the end of EVERY epoch to prevent losing progress
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        epoch_save_path = f"{os.path.splitext(save_path)[0]}_epoch{epoch+1}.pth"
        torch.save(model.state_dict(), epoch_save_path)
        print(f"--> Saved checkpoint: {epoch_save_path} and updated {save_path}")

        # Step learning rate scheduler if present
        if scheduler:
            scheduler.step()
            print(f"Current Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        # Clean memory between epochs to ensure Windows/DirectML driver stability
        gc.collect()

    print(f"\nTraining complete. Final weights saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="IronMan Gesture Model Trainer")
    parser.add_argument('--stream', type=str, choices=['2d', '3d'], required=True,
                        help="Which data stream to train: '2d' for HaGRID, '3d' for Jester")
    parser.add_argument('--epochs', type=int, default=2,
                        help="Number of full passes over the dataset (default: 2)")
    parser.add_argument('--lr', type=float, default=0.0001,
                        help="Learning rate for Adam optimizer (default: 0.0001 for fine-tuning)")
    parser.add_argument('--pool-mode', type=str, choices=['avg', 'flatten'], default='flatten',
                        help="Temporal pooling strategy: 'flatten' (preserves left/right order) or 'avg'")
    parser.add_argument('--dropout', type=float, default=0.5,
                        help="Dropout probability before the FC head (default: 0.5)")
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help="L2 weight decay for the Adam optimizer (default: 1e-4)")
    parser.add_argument('--batch-size', type=int, default=4,
                        help="Batch size for DataLoader (default: 4)")
    parser.add_argument('--train-layer4', action='store_true', default=False,
                        help="Unfreeze ResNet50 layer4 (default: False for fast, crash-free DirectML training)")
    parser.add_argument('--max-samples-per-class', type=int, default=4000,
                        help="Max images per class for balanced 2D HaGRID training (default: 4000)")
    parser.add_argument('--resume', type=str, default=None,
                        help="Path to an existing checkpoint (.pth) to resume training from")
    parser.add_argument('--class-weights', nargs=4, type=float, default=[2.5, 2.5, 1.8, 0.8],
                        help="Loss weights for [Left, Right, No gesture, Doing other things]. Default for fine-tuning: 2.5 2.5 1.8 0.8")
    parser.add_argument('--spatial-zones', type=int, default=3,
                        help="Number of horizontal spatial zones: 3 (Left, Center, Right) or 1 (legacy 1x1 pool)")
    parser.add_argument('--save-name', type=str, default='jester_lstm_spatial',
                        help="Base filename for saved weights in ../weights/ (default: 'jester_lstm_spatial')")
    args = parser.parse_args()

    # Hardware routing
    device = get_device()

    if args.stream == '2d':
        print("Initializing 2D Pipeline (HaGRID)...")
        data_dir = resolve_path("../../data/raw/hagrid")
        batch_size = args.batch_size if args.batch_size > 4 else 32
        dataloader, classes = get_hagrid_dataloader(
            data_dir=data_dir,
            batch_size=batch_size,
            max_samples_per_class=args.max_samples_per_class
        )
        target_names = classes
        print(f" -> Classes: {classes}")

        model = HaGRIDResNet(num_classes=5, train_layer4=args.train_layer4).to(device)

        # Default to resuming from hagrid_resnet50.pth if resume not specified
        default_resume = resolve_path("../weights/hagrid_resnet50.pth")
        resume_target = args.resume or (default_resume if os.path.exists(default_resume) else None)
        if resume_target and os.path.exists(resume_target):
            print(f"Resuming weights from: {resume_target}")
            model.load_state_dict(torch.load(resume_target, map_location=device, weights_only=False))

        params = [{'params': model.backbone.fc.parameters(), 'lr': args.lr}]
        if args.train_layer4:
            params.append({'params': model.backbone.layer4.parameters(), 'lr': args.lr * 0.1})
            print(f" -> Fine-tuning layer4 with lr: {args.lr * 0.1}")

        optimizer = optim.Adam(params, weight_decay=args.weight_decay)
        save_base = 'hagrid_resnet50_finetuned' if args.save_name == 'jester_lstm_spatial' else args.save_name
        save_path = resolve_path(f"../weights/{save_base}.pth")
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    elif args.stream == '3d':
        print("Initializing 3D Pipeline (Jester CNN-LSTM)...")
        csv_path = resolve_path("../../data/raw/jester/Train.csv")
        frames_dir = resolve_path("../../data/raw/jester/Train")

        jester_max = args.max_samples_per_class if args.max_samples_per_class != 4000 else 1800
        dataloader = get_jester_dataloader(
            csv_path=csv_path,
            frames_dir=frames_dir,
            batch_size=args.batch_size,
            max_samples_per_class=jester_max
        )
        target_names = ["Swiping Left", "Swiping Right", "No gesture", "Doing other things"]
        print(f" -> Found {len(dataloader.dataset)} active swipe gestures (batch_size={args.batch_size}).")

        print(f"Instantiating JesterSequenceLSTM model "
              f"(pool_mode='{args.pool_mode}', spatial_zones={args.spatial_zones}, dropout={args.dropout}, train_layer4={args.train_layer4})...")
        model = JesterSequenceLSTM(
            num_classes=4,
            pool_mode=args.pool_mode,
            dropout=args.dropout,
            train_layer4=args.train_layer4,
            spatial_zones=args.spatial_zones
        ).to(device)

        if args.resume and os.path.exists(args.resume):
            print(f"Resuming weights from: {args.resume}")
            state = torch.load(args.resume, map_location=device, weights_only=False)
            model.load_state_dict(state, strict=True)
            print("Successfully loaded model checkpoint!")

        # Optimizer parameters
        params = [
            {'params': model.lstm.parameters()},
            {'params': model.fc.parameters()}
        ]
        if args.train_layer4:
            params.insert(0, {'params': model.cnn[7].parameters(), 'lr': args.lr * 0.1})

        optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
        save_path = resolve_path(f"../weights/{args.save_name}.pth")

        # Class weights: Balances swipe detection with false-positive suppression on 'Doing other things'
        class_weights = torch.tensor(args.class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f" -> CrossEntropyLoss class weights: {class_weights.tolist()}")

    # Launch Training
    train_model(model, dataloader, criterion, optimizer, args.epochs, device, save_path, scheduler=scheduler, target_names=target_names)


if __name__ == "__main__":
    main()