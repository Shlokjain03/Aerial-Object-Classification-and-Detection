from ultralytics import YOLO
import os

def main():
    # 1. Define path to the config file
    # We use absolute path to be 100% sure
    yaml_path = os.path.abspath("data/object_detection_Dataset/data.yaml")
    
    print(f"Loading configuration from: {yaml_path}")

    # 2. Initialize the Model
    # 'yolov8n.pt' will automatically download if you don't have it
    model = YOLO("yolov8n.pt") 

    # 3. Train the Model
    print("Starting YOLOv8 training...")
    results = model.train(
        data=yaml_path,
        epochs=10,      # 10 epochs is enough for a demo. Increase to 50 for better accuracy.
        imgsz=640,      # Standard image size for YOLO
        batch=8,        # Lower batch size to save memory on laptop
        project="models",     # Where to save results
        name="yolov8_drone_bird", # Folder name for this run
        device="cpu"    # Forces CPU (Change to 0 if you have a GPU installed)
    )

    # 4. Validate the Model
    print("\nValidating model on validation set...")
    metrics = model.val()
    print(f"mAP50-95: {metrics.box.map}") # This is the main accuracy score

    # 5. Export/Save Info
    best_weight_path = os.path.join("models", "yolov8_drone_bird", "weights", "best.pt")
    print(f"\nTraining Complete! Best model saved at: {best_weight_path}")

if __name__ == "__main__":
    main()