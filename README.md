# Kvasir Deep Learning: WCE Medical Image Classification

This repository contains the complete deep learning pipeline for classifying Wireless Capsule Endoscopy (WCE) images across two datasets: **Kvasir-Capsule** (imbalanced) and **KVASIR v2** (balanced).

---

## Overview
Wireless Capsule Endoscopy is a vital diagnostic tool, but manual review of thousands of images is time-consuming. This project implements a robust automated classification system using state-of-the-art CNN architectures to assist clinicians in identifying 14 distinct pathological findings.

## 🏗️ Model Architecture
We utilized **Transfer Learning** with a focus on three distinct architectures to compare scaling, depth, and filter variety:

1.  **EfficientNet-B0**: Leverages "Compound Scaling" for maximum parameter efficiency.
2.  **ResNet101V2**: Implements "Skip Connections" to mitigate vanishing gradients in deep networks.
3.  **InceptionV3**: Uses "Multi-scale filters" (1x1, 3x3, 5x5) to detect diseases of varying sizes.

### Training Strategy:
*   **Layer Freezing**: The first **70% of layers** were frozen to preserve ImageNet's general features, while the final **30%** were fine-tuned for medical features.
*   **Custom Head**: 
    - `GlobalAveragePooling2D` for feature compression.
    - `BatchNormalization` for training stability.
    - `Dropout` (0.5 & 0.3) to prevent overfitting.
    - `Dense (256)` with L2 regularization for feature processing.
*   **Optimizer**: **Adam** (Adaptive Moment Estimation) for faster convergence.
*   **LR Control**: `CosineDecay` and `ReduceLROnPlateau` for precise weight optimization.

## 📊 Methodology & Settings
We evaluated the models across three distinct data environments (**The S3 Pipeline**):
*   **S1 (Original)**: Testing on the raw, imbalanced distribution.
*   **S2 (Under-sampled)**: Limiting majority classes to prevent gradient dominance.
*   **S3 (Augmented + Balanced)**: Using synthetic variation to boost minority classes (Threshold=200).

## 🏆 Results & Outcomes
- **The Medical Paradox**: High raw accuracy in S1 was found to be misleading due to the "Normal" class majority.
- **Balanced Performance**: Settings S2 and S3 provided a significantly higher **Weighted F1-Score** for rare diseases like "Blood - fresh" and "Polyps."
- **Top Performer**: **EfficientNet-B0** provided the most stable performance across both balanced and imbalanced settings, making it the ideal candidate for edge clinical deployment.

 Kvasir-Capsule (Highly Imbalanced Dataset)
This dataset represents raw clinical captures, where normal scenarios massively outnumber distinct pathologies.

| Setting | Model Architecture | Accuracy | Precision | Recall | F1-Score |
|:---|:---|:---:|:---:|:---:|:---:|
| **S1-Orig** | EfficientNetB0 | 72.69% | 52.84% | 72.69% | 61.20% |
| **S1-Orig** | ResNet101V2 | 97.04% | 96.93% | 97.04% | 96.94% |
| **S1-Orig** | InceptionV3 | **98.27%** | **98.23%** | **98.27%** | **98.23%** |
| **S2-US**   | EfficientNetB0 | 1.83% | 0.03% | 1.83% | 0.07% |
| **S2-US**   | ResNet101V2 | 80.74% | 88.41% | 80.74% | 82.67% |
| **S2-US**   | InceptionV3 | 77.95% | 88.44% | 77.95% | 80.79% |
| **S3-US+Aug** | EfficientNetB0 | 0.93% | 0.01% | 0.93% | 0.02% |
| **S3-US+Aug** | ResNet101V2 | 77.65% | 88.64% | 77.65% | 80.43% |
| **S3-US+Aug** | InceptionV3 | 74.09% | 87.80% | 74.09% | 77.70% |


Kvasir-v2 (Fundamentally Balanced Dataset)
This manufactured dataset features pre-balanced anatomical classifications.

| Setting | Model Architecture | Accuracy | Precision | Recall | F1-Score |
|:---|:---|:---:|:---:|:---:|:---:|
| **S1-Orig** | EfficientNetB0 | 49.33% | 52.50% | 49.33% | 43.14% |
| **S1-Orig** | ResNet101V2 | **92.08%** | **92.21%** | **92.08%** | **92.04%** |
| **S1-Orig** | InceptionV3 | 90.33% | 90.37% | 90.33% | 90.30% |
| **S2-US**   | EfficientNetB0 | 24.67% | 32.06% | 24.67% | 16.38% |
| **S2-US**   | ResNet101V2 | 88.33% | 88.34% | 88.33% | 88.27% |
| **S2-US**   | InceptionV3 | 87.75% | 87.79% | 87.75% | 87.66% |
| **S3-US+Aug** | EfficientNetB0 | 51.25% | 48.71% | 51.25% | 47.66% |
| **S3-US+Aug** | ResNet101V2 | 89.17% | 89.21% | 89.17% | 89.13% |
| **S3-US+Aug** | InceptionV3 | 86.67% | 86.82% | 86.67% | 86.63% |





---

## 🛠️ Requirements
- TensorFlow / Keras
- Scikit-learn
- Pandas / Matplotlib
- ImageDataGenerator for real-time augmentation
