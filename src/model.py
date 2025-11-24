import torch
import torch.nn as nn
from torchvision import models

# 1. Custom CNN Architecture
class CustomCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomCNN, self).__init__()
        
# Block 1: Input (3, 224, 224) -> Output (32, 112, 112)
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
# Block 2: Input (32, 112, 112) -> Output (64, 56, 56)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
# Block 3: Input (64, 56, 56) -> Output (128, 28, 28)
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
# Block 4: Input (128, 28, 28) -> Output (256, 14, 14)
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

# Flatten and Fully Connected Layers
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(256 * 14 * 14, 512), 
            nn.ReLU(),
            nn.Dropout(0.5),               
            nn.Linear(512, num_classes)    
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# 2. Transfer Learning Architecture (ResNet50)
def get_transfer_model(num_classes=2, freeze_weights=True):
    """
    Loads ResNet50 pre-trained model and modifies the final layer.
    
    Args:
        num_classes (int): Number of output classes (2 for binary).
        freeze_weights (bool): If True, only train the final layer.
    """
    print("Loading ResNet50 pre-trained model...")
    
# Load standard ResNet50 weights
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Freeze early layers so we don't destroy pre-learned patterns
    if freeze_weights:
        for param in model.parameters():
            param.requires_grad = False

# Replace the final 'fc' (fully connected) layer
# ResNet50's original input to fc is 2048 features
    num_features = model.fc.in_features 
    
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    return model

# --- Test block to ensure models load correctly ---
if __name__ == "__main__":
    # Test Custom CNN
    print("Testing Custom CNN...")
    model_custom = CustomCNN()
    dummy_input = torch.randn(1, 3, 224, 224) # Fake image batch
    output = model_custom(dummy_input)
    print(f"Custom CNN Output Shape: {output.shape}") 

    # Test Transfer Learning
    print("\nTesting Transfer Learning Model...")
    model_transfer = get_transfer_model()
    output_transfer = model_transfer(dummy_input)
    print(f"ResNet50 Output Shape: {output_transfer.shape}") 