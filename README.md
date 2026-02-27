# SeaClear Underwater Instance Segmentation  
### A Reproducible Mask R-CNN Pipeline for COCO-Based Underwater Object Detection

---

## Table of Contents
- [Abstract](#abstract)
- [Problem Statement](#problem-statement)
- [Dataset Description](#dataset-description)
- [Label Strategy](#label-strategy)
- [Methodology](#methodology)
- [Results](#results)

---
## Abstract

This repository presents a complete end-to-end project for underwater instance segmentation using the COCO annotation format and a Mask R-CNN (ResNet50-FPN) model implemented with PyTorch and Torchvision.

The project goes beyond simply training a neural network. It includes dataset exploration and validation (EDA), class distribution analysis, label restructuring into meaningful superclasses, controlled mitigation of class imbalance, generation of reproducible train/validation/test splits, model training with mixed precision (AMP), and standardized evaluation using COCO metrics.

The goal is to build not just a trained model, but a structured and reproducible computer vision pipeline that reflects practical machine learning engineering workflow — from raw annotated data to evaluated segmentation results and saved checkpoints.

The final outcome is a clean, evaluation-driven segmentation framework that can be reproduced, extended, and improved further.
---

## Problem Statement

Underwater instance segmentation is a technically demanding task due to a combination of environmental, structural, and dataset-related challenges.

Underwater imagery often suffers from visual degradation. Light absorption and scattering lead to color distortion, reduced contrast, blur, and suspended particles in the water. These factors make object boundaries less distinct and increase the difficulty of accurate segmentation.

In addition, many underwater objects are small, partially occluded, or visually similar to the surrounding environment. Marine organisms and debris may occupy only a small portion of the image, requiring the model to learn detailed spatial representations in order to correctly localize and segment them.

Domain variability further complicates the task. Images collected across different locations, depths, and camera systems introduce distribution shifts that can negatively affect generalization performance.

Finally, dataset structure and class balance play a critical role. A reliable dataset must be consistently annotated, structurally coherent, and reasonably balanced across categories. In practice, underwater datasets frequently exhibit severe class imbalance, where dominant biological subclasses significantly outnumber rarer debris or equipment classes. Without deliberate handling, such imbalance can bias model training and reduce sensitivity to underrepresented categories.

Addressing these challenges requires careful dataset inspection, validation of annotation structure, analysis of class distribution, and controlled strategies to ensure stable and interpretable model training.
---

## Dataset Description

This project is based on the [SeaClear Marine Debris Dataset](https://data.4tu.nl/datasets/4f1dff25-e157-4399-a5d4-478055461689), which contains 8,610 underwater images annotated for object detection and instance segmentation.

The dataset includes 40 object categories, covering not only marine litter but also observed animals, plants, and robot components. This diversity makes the dataset suitable for real-world underwater perception tasks.

Images were captured using Remotely Operated Vehicles (ROVs) during field experiments conducted within the SeaClear project. Data was collected across multiple geographic locations, including:

- Bistrina, Croatia  
- Jakljan, Croatia  
- Lokrum, Croatia  
- Slano, Croatia  
- Marseille, France  

All images are resized to a resolution of 1920×1080 pixels.

Annotations are provided in the COCO format (.json), including bounding boxes and polygon-based instance segmentation masks. Images are organized into folders corresponding to unique site-camera pairs, reflecting domain variability across capture conditions.

The dataset structure makes it directly compatible with modern computer vision frameworks such as PyTorch and Torchvision.

However, preliminary analysis revealed several important characteristics:

- Domain variability due to multiple capture locations and camera setups;
- Significant class imbalance (for example it contain few classes with only 8 annotations);
- Variation in object size, including small and partially occluded instances.

These properties make the dataset realistic and challenging, requiring deliberate preprocessing and label strategy decisions before model training.
---
## Label Strategy

The original dataset contains 40 fine-grained categories, including various types of marine litter, biological organisms, plants, and robot components. While this level of detail is valuable, it introduces practical challenges for model training and evaluation.

Preliminary analysis of class distribution revealed significant imbalance. Several biological subclasses (e.g., different marine organisms) dominate the dataset, while certain debris or equipment categories appear relatively rarely. Training directly on the raw fine-grained taxonomy risks biasing the model toward frequent classes and reducing sensitivity to underrepresented categories.

To address this issue, the following label strategy was implemented:

1. **Superclass Mapping**  
   Fine-grained categories were grouped into broader semantic superclasses (e.g., biological nature, plastic, metal, glass, equipment, construction, etc.).  
   This restructuring reduces label sparsity, improves statistical stability during training, and produces more interpretable evaluation results.

2. **Controlled Downsampling of Dominant Subclasses**  
   Dominant biological subclasses were partially downsampled to mitigate severe imbalance.  
   This prevents the model from overfitting to highly frequent categories and encourages learning across a more balanced distribution.

3. **Consistency with COCO Format**  
   After restructuring, new COCO-compatible train/validation/test splits were generated with updated category identifiers corresponding to the defined superclasses.

This approach preserves the realism of the dataset while improving training stability and ensuring that evaluation metrics better reflect model performance across meaningful semantic groups.
---
## Methodology

The project follows a structured end-to-end pipeline, covering data preprocessing, model training, and standardized evaluation.

### 1. Data Preparation and Validation

- Performed exploratory data analysis (EDA) to inspect dataset structure and annotation consistency;
- Verified COCO annotation integrity (images, bounding boxes, segmentation masks);
- Analyzed class distribution and object size variability;
- Identified dominant subclasses and structural imbalance;
- Generated reproducible train/validation/test splits;
- Exported updated COCO JSON files with superclass labels.

### 2. Model Architecture

The baseline model used in this project is:

**Mask R-CNN with ResNet50-FPN backbone**, implemented via `torchvision.models.detection`.

The architecture includes:

- A Feature Pyramid Network (FPN) for multi-scale feature extraction;
- A Region Proposal Network (RPN) for candidate object region generation;
- Separate heads for bounding box regression/classification and instance mask prediction.

The classification and mask heads were adapted to match the number of defined superclasses.

### 3. Training Setup

Model training was implemented in PyTorch with the following configuration:

- Framework: PyTorch + Torchvision;
- Mixed Precision Training (Automatic Mixed Precision, AMP);
- Gradient accumulation for memory efficiency;
- Optimizer: Stochastic Gradient Descent (SGD);
- Learning rate scheduler: StepLR;
- Batch-based training with reproducible data loading.

Training artifacts include:

- `best.pth` – best performing checkpoint;
- `last.pth` – final checkpoint;
- `metrics.json` – logged evaluation metrics.

### 4. Evaluation Protocol

Model performance was evaluated using the official COCO evaluation protocol via `pycocotools.COCOeval`.

Metrics include:

- Bounding box Average Precision (AP);
- Segmentation mask Average Precision (AP);
- AP at IoU thresholds 0.50 and 0.75.

This standardized evaluation ensures objective comparison and reproducibility of results.
---
## Results

Model performance was evaluated using the official COCO evaluation protocol for both bounding boxes and instance segmentation masks.

Training was conducted for 5 epochs. The best checkpoint was selected based on segmentation AP on the validation set.

### Validation Results (Best Checkpoint)

- **bbox AP (IoU=0.50:0.95)**: 0.474  
- **bbox AP50**: 0.660  
- **bbox AP75**: 0.534  

- **segm AP (IoU=0.50:0.95)**: 0.393  
- **segm AP50**: 0.589  
- **segm AP75**: 0.425  

### Test Results (Best Checkpoint)

- **bbox AP (IoU=0.50:0.95)**: 0.480  
- **bbox AP50**: 0.668  
- **bbox AP75**: 0.542  

- **segm AP (IoU=0.50:0.95)**: 0.407  
- **segm AP50**: 0.609  
- **segm AP75**: 0.448  

The results demonstrate stable baseline performance across both detection and segmentation tasks. Bounding box accuracy reaches 0.48 AP on the test set, while instance segmentation achieves 0.41 AP.

Performance varies across object sizes, with higher accuracy for large objects and reduced performance for small instances — a typical behavior in underwater detection scenarios.

Overall, the trained model provides a solid and reproducible baseline for further optimization and experimentation.
---
## Limitations

While the presented pipeline provides a stable and reproducible baseline, several limitations remain.

First, the model was trained for a limited number of epochs (5), which may restrict convergence and final performance. A longer training schedule could potentially improve both detection and segmentation metrics.

Second, the current setup relies on a standard Mask R-CNN baseline without advanced augmentation strategies or class-balanced loss modifications. More aggressive augmentation policies or imbalance-aware training techniques (e.g., focal-style losses or class-balanced sampling) could further improve performance on rare categories.

Third, performance on small objects remains noticeably lower compared to medium and large instances. This behavior is typical for instance segmentation models, especially in underwater scenes where small objects may be visually ambiguous.

Fourth, no explicit domain adaptation techniques were applied, despite the presence of multi-location and multi-camera data. Cross-domain robustness could be further explored in future work.

These limitations do not undermine the validity of the baseline but instead highlight clear directions for structured improvement.
---
## Future Work

The current implementation establishes a solid and reproducible baseline. Several directions can be explored to further improve performance and robustness.

1. Extended Training Schedule  
   Increasing the number of epochs and applying a more adaptive learning rate policy (e.g., cosine decay) could improve convergence and final AP metrics.

2. Advanced Data Augmentation  
   Incorporating stronger augmentation techniques tailored to underwater imagery (color correction, contrast variation, blur simulation) may improve generalization.

3. Imbalance-Aware Training  
   Applying class-balanced sampling strategies or alternative loss formulations (e.g., focal-style losses) could enhance performance on rare categories.

4. Modern Segmentation Architectures  
   Evaluating newer architectures such as Mask2Former or transformer-based detectors may provide performance improvements over the baseline Mask R-CNN.

5. Domain Generalization  
   Investigating domain adaptation or domain generalization methods to improve cross-location robustness.

6. Inference Optimization  
   Developing a lightweight inference pipeline or deployment-ready version for real-time or near-real-time underwater applications.

These directions provide a clear roadmap for evolving the project from a strong baseline toward a more production-ready segmentation system.
---
## How to Run

1. Download the [SeaClear Marine Debris Dataset](https://data.4tu.nl/datasets/4f1dff25-e157-4399-a5d4-478055461689).

2. Place the dataset inside the `data/` directory of this repository;
   
3. Run the notebooks in order:
   - `SeaClear_EDA_and_splits.ipynb`
   - `SeaClear_train.ipynb`
