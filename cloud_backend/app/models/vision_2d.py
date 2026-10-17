import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class HaGRIDResNet(nn.Module):
    def __init__(self, num_classes: int = 5, train_layer4: bool = False, pretrained: bool = True):
        """
        Initializes the transfer learning architecture.
        num_classes: 5 (hand_heart, mute, no_gesture, take_picture, thumb_index2)
        train_layer4: If True, unfreezes layer4 of ResNet50 for fine-tuning fine spatial finger details.
        pretrained: If True, loads ImageNet backbone weights. Pass False for inference checkpoints.
        """
        super().__init__()
        self.train_layer4 = train_layer4
        
        # 1. Load the ResNet50 backbone
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = resnet50(weights=weights)
        
        # 2. Freeze the backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 3. Optionally unfreeze layer4 for fine-tuning
        if self.train_layer4:
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True

        # 4. Swap the classification head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep earlier backbone layers and BatchNorm statistics in eval mode
        self.backbone.eval()
        if self.train_layer4:
            self.backbone.layer4.train()
            for m in self.backbone.layer4.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        self.backbone.fc.train()
        return self

    def forward(self, x):
        """
        Passes the image tensor through the network.
        Input shape expected: (Batch_Size, 3, 224, 224)
        """
        return self.backbone(x)

# Local validation block
if __name__ == "__main__":
    # Instantiate the model
    model = HaGRIDResNet(num_classes=5)
    
    # Create a dummy image tensor (1 image, 3 RGB channels, 224x224 pixels)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Run a forward pass to ensure the architecture compiles and dimensions match
    output = model(dummy_input)
    
    print("Architecture verified.")
    print(f"Output shape (Batch Size, Classes): {output.shape}")