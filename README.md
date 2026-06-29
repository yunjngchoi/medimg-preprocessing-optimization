# MedImg Preprocessing Optimization

This repository contains experimental code for finding optimal preprocessing pipelines for medical image classification and segmentation.

The project evaluates how different preprocessing strategies affect model performance and computational efficiency across 2D chest X-ray and 3D abdominal trauma CT tasks.

## Objective

The goal of this project is to identify optimal preprocessing pipelines for representative medical imaging models.

The study includes both classification and segmentation tasks across 2D and 3D imaging settings.

## Tasks

This project includes:

- 2D image classification
- 2D image segmentation
- 3D image classification
- 3D image segmentation

## Datasets

This project uses two 2D chest X-ray datasets and one 3D abdominal trauma CT dataset.

### 2D Chest X-ray Datasets

#### VinDr-CXR

VinDr-CXR is used for 2D chest X-ray classification and segmentation experiments.

It contains DICOM chest radiographs with multi-label abnormality annotations, including 14 abnormal findings and normal cases.

#### NIH Chest X-ray 14

NIH Chest X-ray 14, also referred to as CXR-14, is used as an additional 2D chest X-ray dataset.

It contains frontal chest radiographs with image-level multi-label thoracic disease annotations.

### 3D CT Dataset

#### RSNA Abdominal Trauma CT

RSNA Abdominal Trauma CT is used for 3D abdominal trauma CT classification and segmentation experiments.

It contains abdominal CT volumes from trauma patients with patient-level, injury-level, and organ-level labels.

## Preprocessing Search Space

The project searches for optimal preprocessing pipelines using four major preprocessing axes.

### Axis 1. Intensity Processing

This axis evaluates how image intensity normalization or CT windowing affects model performance.

### Axis 2. Spatial Resolution and Resampling

This axis evaluates how image size, voxel spacing, and patch size affect performance and computational cost.

### Axis 3. Data Augmentation

This axis evaluates the effect of augmentation strength and type during model training.

### Axis 4. Frequency and Edge-Aware Processing

This axis evaluates whether additional edge or frequency-domain information improves model performance.

## Experimental Design

The experiments are organized into three phases.

### Phase A. Main Effects Screening

Each preprocessing axis is varied one at a time while the others are fixed at default settings.

Representative models include:

- nnU-Net
- Swin-UNETR
- EfficientNet-B0 U-Net

All models are trained from scratch with fixed random seeds.

Model training uses early stopping, validation-based threshold selection, and bootstrap confidence intervals.

### Phase B. Interaction Study

After Phase A, the most influential preprocessing axes are selected.

The selected axes are combined to evaluate whether preprocessing choices interact with each other across model architectures.

### Phase C. Optimal Pipeline Evaluation

The best preprocessing pipeline is selected for each task, dataset, and model family.

Final evaluation includes test-set performance evaluation, efficiency profiling, and comparison across model architectures.

## Evaluation Metrics

### Classification Metrics

- Macro, micro, and per-class AUC
- Macro, micro, and per-class F1
- Macro and micro MCC
- Mean average precision

### Segmentation Metrics

- Dice Similarity Coefficient
- 95th-percentile Hausdorff Distance
- Normalized Surface Distance

### Efficiency Metrics

- FLOPs
- Throughput
- Peak GPU memory
- Number of parameters

## File Structure


```text
.
├── model_loader.py          # Model loading utilities
├── intensity.py             # Intensity normalization preprocessing experiments
├── resolution.py            # Image resolution preprocessing experiments
├── augmentation.py          # Data augmentation preprocessing experiments
├── frequency.py             # Frequency-domain preprocessing experiments
├── train.py                 # Main training pipeline
├── evaluation_metrics.py    # Metric calculation and model performance evaluation
└── README.md             # Project overview and usage instructions
