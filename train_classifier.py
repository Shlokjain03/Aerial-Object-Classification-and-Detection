import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from src.dataset import get_data_loaders
from src.model import CustomCNN, get_transfer_model
from src.train import train_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def plot_history(history, save_path="models/training_plot.png"):
    """Plots accuracy and loss graphs"""
    acc = history['train_acc']
    val_acc = history['val_acc']
    loss = history['train_loss']
    val_loss = history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    
# Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Training acc')
    plt.plot(epochs, val_acc, 'ro-', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

# Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Training loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.savefig(save_path)
    print(f"Training graphs saved to {save_path}")

def main():
# --- Configuration ---
    DATA_DIR = "data/classfication_dataset"
    MODEL_SAVE_PATH = "models/bird_drone_classifier.pth"
    BATCH_SIZE = 32
    EPOCHS = 10 
    LEARNING_RATE = 0.001
    
# Ensure models folder exists
    os.makedirs("models", exist_ok=True)

# Detect hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

# 1. Load Data
    print("Loading Data...")
    loaders = get_data_loaders(DATA_DIR, BATCH_SIZE)
    
# 2. Initialize Model (ResNet50)
    print("Initializing ResNet50 Transfer Learning Model...")
    model = get_transfer_model(num_classes=2, freeze_weights=True)
    model = model.to(device)

# 3. Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

# 4. Start Training
    print(f"Starting training for {EPOCHS} epochs...")
    trained_model, history = train_model(model, loaders, criterion, optimizer, device, num_epochs=EPOCHS)

# 5. Save Model
    torch.save(trained_model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    
# 6. Plot Results
    plot_history(history)

# 7. Final Evaluation on TEST Set
    print("\nEvaluating on Test Set...")
    train_loader, val_loader, test_loader, classes = loaders # Unpack loaders
    
    trained_model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = trained_model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

# 8. Print Report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))
    
# Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

if __name__ == "__main__":
    main()