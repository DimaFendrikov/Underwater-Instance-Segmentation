# Underwater Instance Segmentation with Mask R-CNN

## 1. Project Idea

This project focuses on **instance segmentation of complex underwater scenes** using a COCO-style dataset.

Underwater imagery presents unique challenges: low contrast, color distortion, dynamic lighting, cluttered backgrounds, and a mix of object scales. The goal of this project is to build a robust and reproducible instance segmentation pipeline capable of detecting and segmenting various underwater objects in realistic conditions.

The project follows a complete machine learning workflow:
- Dataset analysis and preparation  
- Train/validation/test split  
- Model training  
- COCO-style evaluation  
- Qualitative and quantitative performance analysis  

---

## 2. Challenges and Problem Setting

Underwater visual data is particularly difficult for computer vision models due to:

- **Severe class imbalance** — some categories contain very few instances, while others dominate the dataset  
- **Small object segmentation** — many objects occupy a small portion of the image  
- **Domain variability** — different locations and cameras introduce distribution shifts  
- **Visual noise** — sand, vegetation, water turbidity, and motion blur affect detection quality  

Additionally, instance segmentation is inherently more complex than object detection because it requires precise pixel-level mask prediction.

To ensure fair and realistic evaluation, the dataset was split at the **image level** into train, validation, and test subsets, preventing data leakage.

---

## 3. Methodology

The project uses **Mask R-CNN with a ResNet-50 + FPN backbone**, implemented via `torchvision`.

Key methodological components:

- COCO-format dataset and evaluation protocol  
- Training with validation after each epoch  
- Best checkpoint selection based on **validation segmentation AP**  
- Final evaluation on a completely held-out test set  
- Mixed precision training (AMP) for computational efficiency  
- Confidence and mask thresholding for qualitative visualization  

Model performance is evaluated using standard **COCO metrics**:

- AP@[0.50:0.95] (primary metric)  
- AP50 and AP75  
- Separate analysis for small, medium, and large objects  

---

## 4. Implementation Overview

This repository includes a complete end-to-end pipeline.

### Data Processing
- Exploratory Data Analysis (class distribution, object density, size distribution)  
- Optional superclass grouping to reduce label sparsity  
- Train/validation/test split generation in COCO format  

### Training
- Mask R-CNN training loop  
- Validation after each epoch  
- Automatic checkpoint saving (best and last)  
- Reproducible configuration and logging  

### Evaluation
- COCOeval for both bounding boxes and segmentation masks  
- Independent test set evaluation  
- Qualitative GT vs prediction visualization  

---

## 5. Results

### Validation (best checkpoint)

- **bbox AP@[0.50:0.95]: 0.4737**  
- **segm AP@[0.50:0.95]: 0.3926**  
- segm AP50: 0.589  
- segm AP75: 0.425  

### Test (held-out dataset)

- **bbox AP@[0.50:0.95]: 0.4804**  
- **segm AP@[0.50:0.95]: 0.4071**  
- segm AP50: 0.609  
- segm AP75: 0.448  

The test performance is consistent with validation, indicating stable generalization and no significant overfitting.

As expected, segmentation of small objects remains the most challenging case, which is typical for Mask R-CNN in cluttered environments.

---

## 6. Conclusions

This project demonstrates a complete and reproducible instance segmentation workflow for challenging underwater scenes.

Key takeaways:

- A structured train/validation/test pipeline is essential for reliable evaluation  
- COCO-style metrics provide detailed performance analysis  
- The model generalizes well to unseen data  
- Small-object segmentation remains the primary bottleneck  

The current model serves as a strong baseline. Further improvements could include higher input resolution, longer training, stronger augmentation strategies, and anchor tuning.

---

## 7. How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
