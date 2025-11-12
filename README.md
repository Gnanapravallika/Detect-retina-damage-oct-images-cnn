# Detect-retina-damage-oct-images-cnn
# 👁️ Detecting Retinal Damage from OCT Images using Deep Learning (InceptionV3 and Custom CNN)

## Overview

This repository contains a deep learning project focused on classifying Optical Coherence Tomography (OCT) retinal images to automatically detect and categorize common eye diseases. The goal is to build an accurate and efficient classifier to aid in the initial analysis of OCT scans.

The model classifies images into four distinct categories:
1.  **CNV** (Choroidal Neovascularization)
2.  **DME** (Diabetic Macular Edema)
3.  **DRUSEN**
4.  **NORMAL** (Healthy Retina)

The project explores two different CNN strategies: a lightweight **Custom Sequential CNN** and **Transfer Learning** using the powerful **InceptionV3** architecture.

---

## 💾 Dataset

The project uses a large dataset of OCT images published by Kermany et al. (2018).

* **Total Images:** 84,495 JPEG X-Ray images.
* **Classes:** 4 categories (NORMAL, CNV, DME, DRUSEN).
* **Structure:** The dataset is pre-split into `train`, `test`, and `val` directories.

| Class | Description |
| :--- | :--- |
| **NORMAL** | Normal retina images represent healthy retinal cross-sections. |
| **CNV** | Choroidal Neovascularization (abnormal growth of new blood vessels beneath the retina, often associated with age-related macular degeneration). |
| **DME** | Diabetic Macular Edema (accumulation of fluid in the macula due to diabetic retinopathy). |
| **DRUSEN** | Small yellowish/whitish deposits (early signs of diseases like AMD). |

---

## 🛠️ Project Setup

### Technologies Used

* **Framework:** TensorFlow / Keras
* **Language:** Python
* **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, `cv2`, `ImageDataGenerator` (for augmentation), `InceptionV3`.

### Prerequisites

You can install most dependencies using `pip`:

bash
pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn pillow opencv-python

Data Preparation
Images were consistently resized to (299, 299) for model input and processed as grayscale (1 channel).

Data Augmentation (Applied during training):

rotation_range=20

width_shift_range=0.20

height_shift_range=0.15

zoom_range=0.2

horizontal_flip=True

🚀 Model Architectures & Results
1. Model 1: Custom Sequential CNN
A simple Sequential model used Convolutional and MaxPooling layers followed by a single Dense layer.

Metric,Test Loss,Test Accuracy
Score,0.8123,0.7303

Classification Report (Test Set):
Class,Precision,Recall,F1-Score
CNV (0),0.82,0.88,0.85
DME (1),0.61,0.54,0.57
DRUSEN (2),0.74,0.72,0.73
NORMAL (3),0.77,0.80,0.78
Average Accuracy,,,0.73

2. Model 2: InceptionV3 Transfer Learning (Feature Extractor)
The InceptionV3 network, pre-trained on ImageNet, was used with its weights frozen (include_top=False), and custom Dense layers were added for classification.

Input Handling: Grayscale images were converted to 3-channel input for InceptionV3.

Optimizer: Adam (learning_rate=0.0001)

Metric,Test Loss,Test Accuracy
Score,0.5702,0.8355

Classification Report (Test Set):
Class,Precision,Recall,F1-Score
CNV (0),0.90,0.95,0.93
DME (1),0.70,0.74,0.72
DRUSEN (2),0.90,0.88,0.89
NORMAL (3),0.74,0.77,0.75
Average Accuracy,,,0.83

Model 3: InceptionV3 Fine-Tuning (Regularized)
This fine-tuned version utilized a lower learning rate, L2 regularization, and Dropout to optimize the model further and reduce overfitting.

Regularization: L2 regularization (l2(0.01)) on Dense layers.

Optimizer: Adam (learning_rate=1e-5).

Callbacks: EarlyStopping (patience=5).

Metric,Test Loss,Test Accuracy
Score,0.5739,0.8360
Classification Report (Unseen Validation Set):
Class,Precision,Recall,F1-Score,Support
CNV (0),0.82,0.88,0.85,32
DME (1),0.83,0.75,0.79,32
DRUSEN (2),0.81,0.72,0.76,32
NORMAL (3),0.76,0.87,0.81,32
Average Accuracy,,,0.80,128

Saved Model
The final InceptionV3 Fine-Tuning model is saved as InceptionV3_tuning.keras and can be loaded directly for predictions.
