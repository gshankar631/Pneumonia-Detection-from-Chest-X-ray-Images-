Enhanced Pneumonia Detection from Chest X-ray Images Using ResNet50 and Gradient-based Visual Explanations (Grad-CAM)
Project Overview
Pneumonia remains a leading cause of illness and death worldwide, particularly among children under five and the elderly. Manual interpretation of chest X-rays is time-consuming, error-prone, and dependent on expert radiologists. This project presents an AI-powered pneumonia detection system that combines deep learning accuracy with interpretability to support clinical decision-making.
Two models were developed and evaluated:
1. Fine-tuned ResNet50 leveraging transfer learning for high accuracy.
2. Custom lightweight CNN designed for resource-limited deployment.
To ensure clinical reliability and transparency, Grad-CAM visualisations were integrated, highlighting lung regions that most influenced model decisions.

Objectives
Develop and fine-tune a ResNet50 model for pneumonia detection.
Design a lightweight custom CNN to balance accuracy and efficiency.
Implement custom preprocessing and augmentation pipelines for dataset balancing and feature enhancement.
Evaluate models using metrics including Accuracy, Precision, Recall, F1-score, ROC-AUC, and Confusion Matrices.
Employ Grad-CAM for model interpretability and visual explanations.
Deploy a web-based interface for real-time prediction and visualization.

Dataset
Source: Publicly available chest X-ray dataset (Dinçer, 2019).
Total Images: 5,232
NORMAL: 1,349
PNEUMONIA: 3,883
Split:
Training: 72.25%
Validation: 12.75%
Testing: 15%

Preprocessing:
Grayscale-to-RGB conversion
Histogram Equalization
Resizing to 224×224
Normalization (mean=0.5, std=0.5)

Preprocessing & Data Augmentation
Standard Pipeline
Resize → RGB Conversion → Rotation (±10°) → Horizontal Flip
Custom Pipeline
Grayscale → RGB Conversion
Histogram Equalization
Gaussian Noise Injection
Random Cropping
Rotation (±10°)
Horizontal Flip
Custom preprocessing was implemented in pure Python for fine control and reproducibility.

Model Architectures
1. ResNet50 (Transfer Learning)
Pre-trained on ImageNet
Final fully connected layer replaced with 2 output neurons (NORMAL, PNEUMONIA)
Loss: Cross-Entropy
Optimizer: Adam (lr=1e-4)
Regularisation: Dropout (p=0.5), Early Stopping
2. Custom CNN
4 Convolutional layers + MaxPooling
Dropout + Batch Normalisation
Dense layers for classification
Lightweight and suitable for deployment on low-resource hardware
 Training Configuration
Framework: PyTorch
Environment: Google Colab (GPU)
Batch Size: 32
Epochs: 20
Optimizer: Adam
Learning Rate Scheduler: ReduceLROnPlateau
Loss Function: CrossEntropyLoss
Augmentation Library: Custom Python functions + torchvision

Results Summary
Model	Preprocessing	Accuracy	Precision	Recall	F1-score	ROC-AUC
ResNet50	Standard	98.1%	97.9%	98.0%	98.0%	0.991
ResNet50	Custom	97.5%	97.2%	97.3%	97.2%	0.986
Custom CNN	Standard	95.4%	95.0%	94.9%	94.9%	0.964
Custom CNN	Custom	94.8%	94.6%	94.3%	94.4%	0.957

Key Findings:
ResNet50 with standard preprocessing achieved the best balance across all metrics.
Grad-CAM heatmaps confirmed focus on clinically relevant lung areas.
The custom CNN offered a lightweight alternative for embedded or low-resource use cases.

 Interpretability with Grad-CAM
Grad-CAM (Gradient-weighted Class Activation Mapping) was implemented to visualize feature attention within the models.
Generated heatmaps overlay lung regions most responsible for classification.
Improved transparency and clinical trust by confirming model focus on relevant structures.



Example output (illustrative):
/outputs/
 ├── pneumonia_gradcam.png
 ├── normal_gradcam.png

Web Application
A Flask-based web interface was developed for:
Real-time image upload and classification
Grad-CAM visualization for explainability
Display of prediction confidence and class probabilities
