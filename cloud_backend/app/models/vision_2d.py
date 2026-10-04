import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class HaGRIDResNet(nn.Module):
    def __init__(self, num_classes: int = 5):
        """
        Initializes the transfer learning architecture.
        num_classes: 5 (thumb_index2, take_picture, hand_heart, mute, no_gesture)
        """
        super().__init__()
        
        # 1. Load the pre-trained ResNet50 backbone
        # We use DEFAULT weights to get the most up-to-date ImageNet training
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # 2. Freeze the backbone parameters
        # This locks the weights so they don't update during training. 
        # It saves massive amounts of GPU memory and prevents us from destroying 
        # the model's pre-existing ability to recognize basic visual features.
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 3. Swap the classification head
        # The original ResNet50 outputs 1000 classes. We extract the number of 
        # input features feeding into that final layer (2048), drop the old layer, 
        # and attach a new Linear layer designed specifically for our 5 classes.
        # By default, new layers have requires_grad=True, meaning ONLY this layer will train.
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

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