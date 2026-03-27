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

---

## 🛠️ Requirements
- TensorFlow / Keras
- Scikit-learn
- Pandas / Matplotlib
- ImageDataGenerator for real-time augmentation
