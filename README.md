# CNN-Projects
Benchmark of CNN architectures (DenseNet121, MobileNetV2, ResNet50, VGG16) for MYC status prediction in Diffuse Large B-Cell Lymphoma from H&amp;E whole-slide images. Includes preprocessing, stain normalization, patch extraction, standardized training, evaluation metrics, confidence intervals, and Grad-CAM interpretability.

# Efficient CNN Benchmarking for MYC Status Prediction in DLBCL Histopathology

This repository contains the implementation supporting the manuscript:

**Efficient CNNs for MYC-Status Classification in Diffuse Large B-Cell Lymphoma Histopathology: A Comparative Benchmark and Deployment Trade-Offs**

---

## Overview

Diffuse Large B-Cell Lymphoma (DLBCL) is a heterogeneous malignancy in which MYC overexpression is associated with aggressive disease progression and poor prognosis. Determining MYC status typically requires immunohistochemistry (IHC) or molecular testing, which may not always be accessible in resource-limited settings.

This project investigates whether morphological features present in routine Hematoxylin and Eosin (H&E) stained whole-slide images (WSIs) can be used to predict MYC status using deep learning.

We benchmark four widely used convolutional neural network (CNN) architectures:

- DenseNet121
- MobileNetV2
- ResNet50
- VGG16

The study evaluates predictive performance, computational efficiency, and model interpretability under a standardized experimental framework.

---

## Key Contributions

- Standardized benchmark of four CNN architectures for MYC prediction from H&E WSIs
- Controlled comparison of performance–efficiency trade-offs
- Patch-level supervised learning framework with WSI-level data splitting
- Confidence interval estimation for robust performance interpretation
- Grad-CAM visualization for model interpretability
- Reproducible pipeline for digital pathology research

---

## Dataset

- 120 whole-slide images (WSIs)
- 60 MYC-positive cases
- 60 MYC-negative cases
- Digitized at 40× magnification (~0.25 µm/pixel resolution)
- Total extracted patches: 463,200
- Patch size: 256 × 256 pixels
- Dataset split ratio: 80% training, 10% validation, 10% testing (WSI-level split)

Ethical approval:
USM/JEPeM/22110749

---

## Methodology

### Preprocessing Pipeline

1. Whole-slide image loading
2. Background removal
3. Macenko stain normalization
4. Patch extraction (256×256 pixels)
5. Tissue filtering (>5% foreground)
6. Dataset stratification at WSI level

---

### Model Training

All models were trained under identical conditions:

| Parameter | Value |
|----------|------|
| Optimizer | Adam |
| Loss | Binary Cross-Entropy |
| Batch size | 64 |
| Epochs | 30, 50, 100, 150, 200 |
| Framework | PyTorch 2.2 |
| Python version | 3.10 |
| GPU | NVIDIA Tesla T4 |

---

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score (macro)
- Specificity
- Confidence Interval (95%)
- Inference time per step

---

## Results Summary

| Model | Accuracy | Macro F1 | Characteristics |
|------|---------|----------|----------------|
| MobileNetV2 | 80.73% | 0.8072 | Best performance-efficiency balance |
| DenseNet121 | 74.48% | strong recall | Strong MYC+ sensitivity |
| ResNet50 | 65.62% | lower generalization | Overfitting observed |
| VGG16 | 67.50% | lower generalization | High complexity vs dataset size |

MobileNetV2 demonstrated the best balance between predictive performance and computational efficiency, supporting potential deployment in digital pathology workflows.

---

## Explainability

Grad-CAM visualization was applied to highlight morphological regions contributing to model predictions, enabling qualitative interpretation of learned histopathological features.

---

## Acknowledgements

Supported by:

1. Fundamental Research Grant Scheme (FRGS), Ministry of Higher Education Malaysia
2. Universiti Sains Malaysia RU Top Down Grant

---

## Contact

- Corresponding Author:
- Chee Chin Lim

- Universiti Malaysia Perlis
- cclim@unimap.edu.my
