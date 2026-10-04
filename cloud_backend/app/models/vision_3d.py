import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights

class JesterResNet3D(nn.Module):
    def __init__(self, num_classes: int = 3):
        """
        Initializes the 3D transfer learning architecture.
        num_classes: 3 (Swiping Left, Swiping Right, no_gesture)
        """
        super().__init__()
        
        # 1. Load the pre-trained ResNet3D 18-layer backbone
        self.backbone = r3d_18(weights=R3D_18_Weights.DEFAULT)
        
        # 2. Freeze the backbone parameters to retain temporal feature extraction
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 3. Swap the classification head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        Passes the video sequence tensor through the network.
        
        Crucial Tensor Manipulation:
        Your DataLoader outputs shape: (Batch, Frames, Channels, Height, Width)
        PyTorch 3D CNNs expect shape:  (Batch, Channels, Frames, Height, Width)
        We use .permute() to swap the Frames (1) and Channels (2) dimensions.
        """
        x = x.permute(0, 2, 1, 3, 4)
        return self.backbone(x)

# Local validation block
if __name__ == "__main__":
    # Instantiate the model for our 3 target classes
    model = JesterResNet3D(num_classes=3)
    
    # Create a dummy video tensor (1 batch, 15 frames, 3 RGB channels, 224x224 pixels)
    # Note: This matches the raw output shape from your dataset.py before permutation
    dummy_sequence = torch.randn(1, 15, 3, 224, 224)
    
    # Run a forward pass
    output = model(dummy_sequence)
    
    print("3D Architecture verified.")
    print(f"Output shape (Batch Size, Classes): {output.shape}")