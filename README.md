# 🌸 Flower Classification with CNN (PyTorch)

This project trains a Convolutional Neural Network (CNN) to classify flower images into multiple classes using PyTorch. The workflow includes custom dataset loading, preprocessing, model training, and evaluation.

## 🎯 Project Objective

Build a reliable multi-class image classifier that can learn visual patterns from flower images (such as edges, textures, and shapes) and predict the correct flower category.

## 📂 Dataset

The dataset can be downloaded using `kagglehub`:

```python
import kagglehub

path = kagglehub.dataset_download("marquis03/flower-classification")
print(path)
```

### Expected structure

```text
root/
├── train/
│   ├── daisy/
│   ├── rose/
│   ├── sunflower/
│   ├── tulip/
│   └── ...
```

Each class should be stored in its own folder.

## 🧠 Problem Type

- **Task:** Multi-class image classification
- **Input:** Flower image
- **Output:** Predicted flower class

## 🏗️ Model Architecture

The model uses a 4-block CNN pipeline. Each block contains:

- Convolution (`Conv2D`)
- Batch Normalization
- ReLU activation
- MaxPooling

Filter progression:

`32 → 64 → 128 → 256`

After convolutional feature extraction, the model uses fully connected layers for final class prediction.

## ⚙️ Why These Components?

- **CNN:** Learns hierarchical visual features automatically.
- **ReLU:** Adds non-linearity and helps efficient training.
- **BatchNorm:** Stabilizes learning and improves convergence.
- **MaxPooling:** Reduces spatial size while preserving important features.
- **Dropout:** Helps reduce overfitting.

## 📦 Custom Dataset Support

The project uses a custom PyTorch `Dataset` class to:

- Load images from class-wise folders
- Ignore invalid/non-image files
- Apply transforms safely
- Return `(image, label)` pairs

## 🔁 Training Setup (Typical)

- Loss function: `CrossEntropyLoss`
- Optimizer: `Adam`
- Batch size: `32`
- Learning rate: `0.001`
- Epochs: `5-10` (adjust as needed)

## 🚀 Training Workflow

1. Load and preprocess images
2. Pass images through CNN
3. Compute predictions and loss
4. Backpropagate gradients
5. Update model weights
6. Track training/validation performance

## 📊 Evaluation

Primary metric:

`Accuracy = Correct Predictions / Total Predictions`

## 🖼️ Visualization

Matplotlib is used to visualize random samples to validate:

- Correct image loading
- Label mapping
- Preprocessing pipeline

## ✅ Key Features

- Custom dataset pipeline
- Robust image filtering
- 4-block CNN architecture
- Batch normalization + dropout
- Multi-class classification
- Kaggle dataset integration
- CUDA/GPU support (if available)

## 🔮 Future Improvements

- Confusion matrix
- Training vs. validation curves
- Transfer learning (ResNet, VGG, EfficientNet)
- Model checkpoint saving (`.pth`)
- Better data augmentation
- Web app demo (Streamlit/Flask)

## 👨‍💻 Author

Developed as a deep learning assignment by **Md. Rifat Islam Rizvi**.
