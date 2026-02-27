# SeaClear Underwater Instance Segmentation  
### A Reproducible Mask R-CNN Pipeline for COCO-Based Underwater Object Detection

---

## Table of Contents
- [Abstract](#abstract)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Dataset Description](#dataset-description)
- [Methodology](#methodology)
- [Results](#results)

---
## Abstract

This project presents a reproducible end-to-end pipeline for underwater instance segmentation built around the COCO annotation format and a Mask R-CNN (ResNet50-FPN) baseline.

Rather than focusing solely on training a model, the primary goal is to design a structured and transparent workflow that reflects real-world machine learning engineering practices. The pipeline includes dataset exploration, class distribution analysis, label restructuring, imbalance mitigation, standardized evaluation using COCO metrics, and qualitative validation of predictions.

Underwater imagery poses distinct challenges: reduced visibility, color distortion, environmental noise, domain variability across locations and cameras, and significant class imbalance. These factors make naïve training setups unreliable and difficult to generalize.

To address these issues, the project introduces a superclass labeling strategy and controlled downsampling of dominant biological subclasses, enabling more stable training and interpretable evaluation.

The final result is not just a trained segmentation model, but a clean, reproducible, and evaluation-driven framework designed to demonstrate robust computer vision engineering methodology.
---

## Problem Statement

Underwater instance segmentation presents multiple technical challenges:

1. **Visual degradation** – color distortion, low contrast, blur, and suspended particles significantly reduce object clarity;
2. **Small and partially occluded objects** – debris and marine organisms often occupy small regions or blend with the background;
3. **Domain variability** – multiple capture locations and camera systems introduce distribution shifts;
4. **Severe class imbalance** – dominant biological subclasses may bias model training and suppress rare but important categories.

A naïve training approach can lead to:

- Overfitting to dominant classes;
- Poor cross-domain generalization;
- Misleading evaluation results;
- Low reproducibility.

Therefore, the task requires a structured data pipeline, a deliberate label strategy, and standardized evaluation metrics to ensure robust and interpretable results.

---

## Objectives

### Main Objective

Develop a reproducible and evaluation-driven pipeline for underwater instance segmentation using COCO-formatted annotations.

### Technical Objectives

- Perform dataset exploration and structural validation (EDA);
- Analyze class distribution and identify imbalance issues;
- Introduce superclass mapping for robust label representation;
- Apply controlled downsampling to dominant subclasses;
- Generate consistent COCO train/val/test splits;
- Train a Mask R-CNN (ResNet50-FPN) baseline;
- Evaluate performance using COCOeval for both bounding boxes and segmentation masks;
- Provide qualitative validation via Ground Truth vs Prediction visualization;
- Store reproducible artifacts (splits, checkpoints, metrics logs).
