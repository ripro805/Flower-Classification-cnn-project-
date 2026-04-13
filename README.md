# 🌸 Flower Classification using CNN (PyTorch + Custom Dataset)

This project is a deep learning solution for multi-class flower image classification using a Convolutional Neural Network (CNN) built with PyTorch. The dataset is loaded using a fully custom Dataset class to ensure flexibility and better understanding of image preprocessing pipelines.

---

# 📌 Project Objective

The goal of this project is to classify flower images into multiple categories by training a deep CNN model that learns visual patterns such as edges, textures, shapes, and complex flower structures.

---

# 📂 Dataset

The dataset is downloaded using KaggleHub:

```python
import kagglehub

path = kagglehub.dataset_download("marquis03/flower-classification")
print(path)



📁 Expected Structure

root/
├── train/
│ ├── daisy/
│ ├── rose/
│ ├── sunflower/
│ ├── tulip/
│ └── ...
└── (or direct class folders)

Each folder represents a unique class label.

🧠 Problem Type
Multi-class image classification
Input: Flower images
Output: Flower category (K classes)
🏗️ Model Architecture (CNN)

The model is a 4-block Convolutional Neural Network.

Each block includes:

Convolution layer (Conv2D)
Batch Normalization
ReLU Activation
MaxPooling
📊 Filter progression:

32 → 64 → 128 → 256

🔬 Why CNN?

CNN is used because it:

Automatically extracts image features
Reduces manual feature engineering
Performs very well on image data
Learns hierarchical representations
⚙️ Activation Function (ReLU)

ReLU(x) = max(0, x)

Benefits:

Adds non-linearity
Helps learn complex patterns
Speeds up training
🧪 Batch Normalization

BatchNorm is used to:

Normalize activations
Stabilize training
Improve convergence speed
Reduce overfitting
📉 MaxPooling

MaxPooling is used to:

Reduce spatial dimensions
Keep important features
Reduce computation cost

Example:
128×128 → 64×64 → 32×32 → 16×16

🧱 Fully Connected Layers

After feature extraction:

Flatten → Dense Layers

Linear(25688 → 512)
ReLU
Dropout(0.5)
Linear(512 → K classes)
🎯 Output Layer

The final layer outputs class probabilities.

Example:

Rose: 0.10
Sunflower: 0.80
Daisy: 0.05
Tulip: 0.05

👉 Highest probability = predicted class

📦 Custom Dataset

A fully custom PyTorch Dataset class is used.

Features:
Loads images from folder structure
Filters only valid image files
Assigns labels automatically
Applies transformations
Returns (image, label)
⚠️ Important Fixes in This Project

This project includes fixes for common errors:

✔ Non-image file filtering (.txt, .csv removed)
✔ Empty dataset handling
✔ Safe train-test split
✔ Robust DataLoader creation
✔ Safe random image visualization

🔁 Training Setup
Loss Function: CrossEntropyLoss
Optimizer: Adam
Epochs: 5–10
Batch Size: 32
Learning Rate: 0.001
🚀 Training Process
Images are loaded through custom dataset
Passed through CNN model
Predictions are generated
Loss is calculated
Backpropagation updates weights
Model gradually improves accuracy
📊 Evaluation

Accuracy is calculated using:

Accuracy = Correct Predictions / Total Predictions

🖼️ Visualization

Random flower images are displayed using Matplotlib to verify dataset loading and preprocessing.

🔄 Workflow

Input Image
→ Conv Block 1 (edges)
→ Conv Block 2 (textures)
→ Conv Block 3 (shapes)
→ Conv Block 4 (complex patterns)
→ Flatten
→ Fully Connected Layer
→ Dropout
→ Output Layer (Prediction)

📈 Key Features
Custom Dataset implementation
Fully error-free data pipeline
4-layer deep CNN architecture
Batch Normalization
Dropout regularization
Multi-class classification
Kaggle dataset integration
GPU support (CUDA)
🔮 Future Improvements
Confusion Matrix
Training vs Validation Graph
Transfer Learning (ResNet, VGG)
Model saving (.pth)
Web deployment using Streamlit/Flask
Data augmentation improvements
👨‍💻 Author

This project is developed as a Deep Learning assignment using PyTorch for multi-class image classification.
