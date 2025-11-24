# 🚁 Aerial Object Classification & Detection

## 📌 Project Overview
This project provides an AI-powered solution for Aerial Surveillance. It is designed to identify and locate automatically **Birds** and **Drones** in aerial imagery. This technology is critical for:
- Preventing **Bird Strikes** at airports.
- Monitoring unauthorized **Drone Activity** in restricted airspace.
- protecting wildlife and ensuring airspace safety.

## 🚀 Features
- **Data Preprocessing:** Custom pipeline for resizing, augmentation, and normalisation.
- **Image Classification:** Uses **ResNet50** (Transfer Learning) to classify images as "Bird" or "Drone" with 98%+ accuracy.
- **Object Detection:** Uses **YOLOv8** to detect objects and draw bounding boxes in real-time.
- **Web Application:** A user-friendly **Streamlit** dashboard for easy testing and visualization.

## 📂 Dataset
The dataset consists of aerial images categorised into Birds and Drones.
Due to file size limits, the dataset is hosted externally.

👉 **[DOWNLOAD DATASET HERE] (https://drive.google.com/drive/folders/1nn1vqsh8juhafkJcleembrjQ9EqtIoMh?usp=sharing) / (https://drive.google.com/drive/folders/114wV_igIhWldcG0HftNIZZsivrs8G22p?usp=sharing)**

## 🛠️ Tech Stack
- **Language:** Python
- **Deep Learning Framework:** PyTorch
- **Object Detection:** Ultralytics YOLOv8
- **Web Interface:** Streamlit
- **Libraries:** OpenCV, PIL, Matplotlib, NumPy

## ⚙️ Installation
1. **Clone the Repository**
   ```bash
   git clone [https://github.com/your-username/aerial-object-detection.git](https://github.com/your-username/aerial-object-detection.git)
   cd aerial-object-detection

2. Install Dependencies
3. Bash:
pip install -r requirements.txt

4. Download Model Weights (Optional) If you do not want to retrain, download the pre-trained weights from the Drive link above and place them in the models/ folder.

🏃‍♂️ Usage

1. Run the Web App (Easiest Way)
To start the interface and test the model:
Bash:
streamlit run app.py

2. Retrain the Classifier
To retrain the ResNet50 classifier on your own data:
Bash:
python train_classifier.py

3. Retrain YOLOv8
To retrain the object detection model:
Bash
python train_yolo.py

📊 Results
Classification Accuracy: ~99% (ResNet50)

Object Detection: High Precision/Recall on test videos.

📁 Project Structure
├── data/                  # Dataset (Not included in repo, see link above)
├── models/                # Trained model files (.pth, .pt)
├── src/
│   ├── dataset.py         # Data loading and augmentation
│   ├── train.py
│   ├── model.py           # CNN and ResNet50 architecture
├── app.py                 # Streamlit Web Application
├── train_classifier.py    # Training script for classification
├── train_yolo.py          # Training script for YOLOv8
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation

👨‍💻 Author
Developed by [Shlok Jain] as part of an Internship Project.

1.  Open your terminal in the project folder.
2.  Run these commands:
```bash
git init
git add .
git commit -m "Initial commit - Aerial Object Detection Project"
