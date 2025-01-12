# Face Recognition with OpenFace and MediaPipe

## Overview
This project implements a face recognition system using TensorFlow, MediaPipe for face detection, and a VGG16-based architecture for facial embedding extraction. The system is designed to train on anchor, positive, and negative face datasets to create embeddings and perform face verification or recognition.

The system includes features such as face cropping, dataset preprocessing, embedding visualization using PCA and t-SNE, and accuracy evaluation.

## Features
- **Face Detection**: Utilizes MediaPipe for precise face detection and cropping.
- **Facial Embedding Generation**: Implements a VGG16-based architecture for extracting embeddings.
- **Triplet Loss**: Trains the model to distinguish between anchor, positive, and negative embeddings.
- **Visualization**: Embedding distributions are visualized with PCA and t-SNE.
- **Accuracy Evaluation**: Measures the model's effectiveness using positive and negative distance distributions.

## Requirements
Ensure you have the following installed:
- Python 3.8 or above
- TensorFlow 2.13
- MediaPipe
- OpenCV
- Matplotlib
- Scikit-learn
- Plotly

Install the required Python packages with:
```bash
pip install tensorflow==2.13 mediapipe opencv-python matplotlib scikit-learn plotly
```

## Directory Structure
```
project/
  |-- face_data/
  |    |-- anchor/
  |    |-- positives/
  |    |-- negatives/
  |-- face_recog_openface_mediapipe.ipynb
  |-- requirements.txt
  |-- README.md
```

- **`anchor/`**: Contains anchor face images.
- **`positives/`**: Contains positive face images (similar to anchor).
- **`negatives/`**: Contains negative face images (different from anchor).
- **`face_recog_openface_mediapipe.py`**: Main Python file for training and evaluating the model.

## How It Works
1. **Face Detection and Cropping**:
   - MediaPipe detects faces and crops the images.
   - Cropped faces are saved for anchor, positive, and negative datasets.

2. **Dataset Preprocessing**:
   - Images are resized to 224x224 and normalized.
   - Anchors, positives, and negatives are aligned to the smallest dataset size.

3. **Model Training**:
   - Uses a VGG16-based architecture.
   - Applies triplet loss to train on anchor, positive, and negative samples.
   - Includes early stopping to optimize training.

4. **Embedding Visualization**:
   - PCA and t-SNE reduce embeddings to 2D for visualization.

5. **Accuracy Evaluation**:
   - Measures positive and negative distances to calculate accuracy.
   - Visualizes the distance distributions.

## Usage
1. Clone the repository:
```bash
git clone https://github.com/Igwe-Ugo/Facial_recognition_system.git
```

2. Navigate to the project directory:
```bash
cd Facial_recognition_system
```

3. Run the main Python script:
```bash
python face_recog_openface_mediapipe.ipynb
```

4. Preprocess datasets and train the model. The script saves the trained model as `face_recog_vggface.keras`.

5. Evaluate the model and visualize embeddings.

## Visualization Outputs
- **PCA**: Visualizes embeddings based on their principal components.
- **t-SNE**: Highlights clusters of anchors, positives, and negatives.
- **Distance Distribution**: Displays positive and negative pair distributions.

## Contributing
Contributions are welcome! Feel free to submit issues or pull requests for improvements.

## Acknowledgments
- **TensorFlow**: For providing tools for deep learning.
- **MediaPipe**: For efficient face detection.
- **VGG16**: For the backbone architecture in this implementation.

