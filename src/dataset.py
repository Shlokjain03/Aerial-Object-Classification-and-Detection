import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Define standard mean and std deviation for Transfer Learning (ResNet/MobileNet)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def get_transforms(stage='train'):
    """
    Returns the image transformation pipeline.
    
    Args:
        stage (str): 'train' for training (includes augmentation), 
                     'val' or 'test' for evaluation (no augmentation).
    """
    if stage == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),# Resize to fixed size
            transforms.RandomHorizontalFlip(p=0.5), 
            transforms.RandomRotation(degrees=15),  
            transforms.ColorJitter(brightness=0.2, contrast=0.2),#Random lighting
            transforms.ToTensor(),# Convert to Tensorfrom 0-1 range
            transforms.Normalize(MEAN, STD)# Normalize
        ])
    else:
# For validation and testing, we cn only resize and normalize
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])

def get_data_loaders(data_dir, batch_size=32):
    
    
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')

# Check if directories exist
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training folder not found at {train_dir}. Check your folder structure.")

# Load Datasets with specific transforms
    train_dataset = datasets.ImageFolder(root=train_dir, transform=get_transforms('train'))
    valid_dataset = datasets.ImageFolder(root=valid_dir, transform=get_transforms('val'))
    test_dataset = datasets.ImageFolder(root=test_dir, transform=get_transforms('test'))

# Create DataLoaders (The iterator that feeds the model)
# Shuffle=True for training so the model doesn't memorize the order
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, valid_loader, test_loader, train_dataset.classes

if __name__ == "__main__":
    print("Testing Data Loading...")
    
# This updates the  path to where actual data is
    TEST_DATA_PATH = "data/classfication_dataset" 
    
    try:
        train_loader, val_loader, _, classes = get_data_loaders(TEST_DATA_PATH)
        print(f"Classes found: {classes}")
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        
# Fetch one batch to inspect dimensions
        images, labels = next(iter(train_loader))
        print(f"Batch Image Shape: {images.shape}") # Should be [32, 3, 224, 224]
        print("Data loading successful!")
        
    except Exception as e:
        print(f"Error: {e}")