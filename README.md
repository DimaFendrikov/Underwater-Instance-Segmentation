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

This project implements a reproducible end-to-end pipeline for **underwater instance segmentation** using the COCO annotation format and a Mask R-CNN (ResNet50-FPN) baseline.

The objective is not limited to training a segmentation model. Instead, the focus is on constructing a structured and evaluation-driven workflow that includes dataset analysis, label restructuring, imbalance mitigation, standardized COCO evaluation, and qualitative validation.

Underwater imagery introduces domain variability, visual degradation, and severe class imbalance. To address these challenges, the pipeline incorporates a superclass labeling strategy and controlled downsampling of dominant biological subclasses.

The result is a clean and reproducible segmentation framework that reflects practical machine learning engineering standards rather than a single experimental run.

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
