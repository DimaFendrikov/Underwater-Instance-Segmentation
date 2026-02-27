# SeaClear Underwater Instance Segmentation – Mask R-CNN (COCO)

End-to-end computer vision pipeline for **instance segmentation** on underwater imagery:  
**EDA → label strategy (superclasses) → COCO splits → training Mask R-CNN → COCOeval → checkpoints → visual validation**.

The project focuses not only on training a model, but on building a **reproducible dataset + evaluation workflow** that looks credible for real-world usage and research-style reporting.

---

## Problem

Underwater vision is challenging in practice:

1. Images often have low visibility (turbidity), color shift, blur, reflections and uneven lighting;
2. Objects can be small, partially occluded and visually similar to the background;
3. A typical dataset contains multiple domains (different locations and cameras), which increases domain shift;
4. Labels are frequently **highly imbalanced** – for example, biological categories may dominate the dataset and push the model to ignore rarer debris/equipment classes.

As a result, a “train once and report accuracy” approach is not enough.  
A strong solution must include **dataset analysis, a clear label strategy, correct COCO evaluation, and qualitative checks**.

---

## Idea and Approach

The core idea is to build a clean, recruiter-ready pipeline that solves two key issues:

1. **Imbalance and noisy taxonomy** – fine-grained labels are remapped into meaningful **superclasses** and dominant bio subclasses are downsampled to stabilize training;
2. **Reproducibility** – the pipeline produces consistent artifacts (splits, checkpoints, metrics) and evaluates the model using the standard **COCOeval** protocol for both bounding boxes and masks.

Technically, the repository implements:

1. Dataset EDA with mask/bbox visualization (sanity checks);
2. Domain-aware indexing (location/camera/domain);
3. Controlled downsampling of dominant biological subclasses;
4. Mapping fine-grained labels into robust **superclasses**;
5. Export of COCO `train/val/test` splits;
6. Training `maskrcnn_resnet50_fpn` (torchvision) with AMP + gradient accumulation;
7. Evaluation with COCOeval (`bbox` and `segm`) + saving `best.pth`, `last.pth` and `metrics.json`;
8. Visualization of **Ground Truth vs Predictions** for qualitative validation.
