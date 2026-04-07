# Complete Analysis: Mask R-CNN Bone Tumor Detection Notebook

## 📋 Executive Summary

This Jupyter notebook implements a **complete end-to-end pipeline** for bone tumor detection and segmentation using Mask R-CNN on the BTXRD (Bone Tumor X-ray Dataset). The notebook consists of **6 major stages** executed in sequential cells, covering data preprocessing, model training, and comprehensive evaluation using multiple metrics.

**Dataset**: 3,746 X-ray images from 1,008 patients across 3 medical centers
**Task**: Binary tumor detection + instance segmentation
**Model**: Mask R-CNN with ResNet-50-FPN backbone
**Framework**: Detectron2 (Facebook AI Research)

---

## 🏗️ Notebook Structure (18 Cells)

### Cell Distribution:
- **Cell 0** (1009 lines): Stage 1 - Data Preprocessing & COCO Conversion
- **Cell 2** (724 lines): Stage 2 - Mask R-CNN Training
- **Cell 4** (852 lines): Size-Stratified Sensitivity Analysis
- **Cell 5** (765 lines): Confusion Matrix Generator
- **Cell 8** (667 lines): Comprehensive AP Metrics Evaluation
- **Cell 10** (1067 lines): Segmentation Metrics Evaluation
- **Cell 12** (834 lines): Image-Level Evaluation (Max Aggregation)
- **Cell 14** (875 lines): Detection-Level Evaluation (Journal Standard)
- **Other cells**: Empty (placeholders)

---

## 📊 STAGE 1: Data Preprocessing & COCO Conversion

### Purpose
Convert LabelMe annotations to COCO format and create scientifically rigorous train/val/test splits that prevent data leakage and control center bias.

### What Was Done

#### 1. **Patient-Level Grouping**
```
Problem: Same patient may have multiple X-ray images (different views)
Solution: Group images by patient using a "patient fingerprint"
```

**Patient Fingerprint Components** (Label-Free):
- Center ID (1, 2, or 3)
- Age (normalized)
- Gender (F/M)
- Anatomy (multiple bones, encoded as binary features)
- Joint involvement (multiple joints, encoded)

**Critical Design Choice**: The fingerprint **excludes** diagnostic labels (tumor/benign/malignant) to prevent label leakage during patient identification.

**Statistics Generated**:
- Total patients: 1,008 (from 3,746 images)
- Average images per patient: 3.72
- Patients with multiple views: 688 (68%)
- Collision prevention: Anatomy features prevented 713 potential ID collisions (241.7%)

#### 2. **Center-Aware Stratified Splitting**

**Why This Matters**: Different medical centers may have different:
- Equipment quality
- Image acquisition protocols
- Patient demographics
- Diagnostic patterns

**Distribution Across Centers**:
```
Center 1: 698 patients (69.2%) - 2,938 images
Center 2: 173 patients (17.2%) - 549 images
Center 3: 137 patients (13.6%) - 259 images
```

**Splitting Strategy**:
1. Create stratification groups: `class × center` (e.g., "Benign_Center1")
2. Use scikit-learn's StratifiedGroupKFold
3. Split ratios: 70% train / 15% validation / 15% test
4. **Key**: All images from same patient stay in same split

**Results**:
- Train: 704 patients, 2,602 images
- Val: 152 patients, 580 images
- Test: 152 patients, 564 images

#### 3. **Class Distribution** (by patient, majority voting)
```
Normal:    218 patients (21.6%)
Benign:    675 patients (67.0%)
Malignant: 115 patients (11.4%)
```

**Important Note**: For Mask R-CNN detection:
- "Normal" images have **zero annotations** (no bounding boxes)
- "Benign" and "Malignant" are **combined** into single "tumor" class
- This creates binary detection: Tumor vs. Normal

#### 4. **COCO Format Conversion**

**Why COCO Format?**
- Standard format for Detectron2
- Supports bounding boxes + segmentation masks
- Includes image metadata

**COCO Components Created**:
```json
{
  "images": [...],  // Image metadata
  "annotations": [...],  // Bounding boxes + RLE-encoded masks
  "categories": [{"id": 1, "name": "tumor"}]  // Single category
}
```

**Annotation Processing**:
- LabelMe polygons → COCO bounding boxes (x, y, width, height)
- Polygons → RLE (Run-Length Encoding) masks
- Each tumor gets unique annotation ID
- Normal images included with empty annotation lists

**Output Files**:
```
preprocessed/
├── coco_annotations/
│   ├── train.json (2,602 images, 1,617 annotations)
│   ├── val.json (580 images, 364 annotations)
│   └── test.json (564 images, 337 annotations)
```

#### 5. **Metadata Preprocessing** (23 Features, NO Center)

**Critical Design**: Center ID is **excluded** from model input features to prevent the model from learning center-specific shortcuts.

**Feature Engineering**:
- Age: Z-score normalization (mean=34.34, std=20.81)
- Gender: Binary encoding
- Anatomy: Multi-hot encoding (15 bone types)
- Joint: Multi-hot encoding (7 joint types)
- Class: One-hot encoding (Normal/Benign/Malignant)

**Total Features**: 23 features per image

#### 6. **Validation & Quality Checks**

✅ **Patient Leakage Check**: Verified no patient appears in multiple splits
✅ **File Alignment**: All images have corresponding metadata
✅ **Class Balance**: Verified proportions maintained across splits
✅ **Center Distribution**: Verified center representation in all splits

### Tools & Libraries Used
- **pandas**: DataFrame operations
- **numpy**: Numerical computations
- **json**: COCO file I/O
- **sklearn.model_selection**: StratifiedGroupKFold for splitting
- **pycocotools**: Mask encoding/decoding
- **tqdm**: Progress bars
- **pathlib**: File path handling

### Critical Methodological Choices for Publication

The code explicitly documents these for research papers:

1. **Patient IDs are approximated** (no explicit identifiers in dataset)
2. **Majority voting** used for multi-image patients
3. **Center excluded** from model inputs (only used for stratification)
4. **Normal images included** for realistic class imbalance

---

## 🚀 STAGE 2: Mask R-CNN Training

### Purpose
Train a Mask R-CNN model for tumor detection and instance segmentation with all critical bugs fixed.

### Architecture Details

**Base Model**: Mask R-CNN (He et al., 2017)
**Backbone**: ResNet-50 with Feature Pyramid Network (FPN)
**Framework**: Detectron2

**Why Mask R-CNN?**
1. **Two-stage detector**: Region Proposal Network (RPN) + RoI classifier
2. **Instance segmentation**: Produces pixel-level masks
3. **High accuracy**: Better than single-stage detectors like YOLO/SSD
4. **Detectron2 support**: Production-ready, well-maintained

### Training Configuration

#### Image Preprocessing
```python
INPUT_SIZE = (800, 1024)  # (height, width)
AUGMENTATIONS:
  - ResizeShortestEdge(640, 640, max_size=912)
  - RandomFlip (horizontal)
  - CLAHE (Contrast Limited Adaptive Histogram Equalization)
```

**CLAHE Processing**:
- **Purpose**: Enhance local contrast in X-rays
- **Parameters**: clipLimit=2.0, tileGridSize=(8,8)
- **Critical Fix**: Handles float images [0,1] correctly
- **Grayscale Detection**: Efficiently processes grayscale-as-BGR

#### Model Hyperparameters
```python
BACKBONE: "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
NUM_CLASSES: 1  # Single "tumor" class
BATCH_SIZE: 2   # Per GPU
LEARNING_RATE: 0.00025
LR_SCHEDULER: "WarmupMultiStepLR"
  - Warmup iterations: 500
  - Steps: [6000, 8000] (learning rate decay points)
  - Gamma: 0.1 (decay factor)

MAX_ITER: 26520 (85 epochs @ 312 iters/epoch)
WEIGHT_DECAY: 0.0001
MOMENTUM: 0.9

RPN_IOU_THRESHOLDS: [0.3, 0.7]
ROI_HEADS:
  - BATCH_SIZE_PER_IMAGE: 512
  - NUM_CLASSES: 1
  - SCORE_THRESH_TEST: 0.15  # Inference threshold
  - IOU_THRESHOLDS: [0.5]
  - NMS_THRESH: 0.5 (Non-Maximum Suppression)
```

#### Hardware & Optimization
```
GPU: NVIDIA Tesla T4
Mixed Precision: Enabled (AMP)
Gradient Clipping: Enabled (max_norm=5.0)
Checkpoint Saving: Every 5 epochs
```

### Critical Bug Fixes Applied

The code explicitly documents **6 critical fixes**:

#### Fix 1: Image Size Mismatch
**Problem**: Training used 800×1024, evaluation used different size
**Solution**: Consistent size for train and eval
**Impact**: Prevents evaluation artifacts

#### Fix 2: CLAHE Conversion
**Problem**: CLAHE expects uint8, but normalized images are float32 [0,1]
**Solution**: Convert float→uint8→CLAHE→float
**Impact**: Proper contrast enhancement

#### Fix 3: Optimizer Rebuild
**Problem**: AMP scaler corrupted after checkpoint load
**Solution**: Rebuild GradScaler when resuming from checkpoint
**Impact**: Stable mixed-precision training

#### Fix 4: Scheduler State
**Problem**: Learning rate reset after optimizer rebuild
**Solution**: Preserve and restore scheduler state
**Impact**: Correct learning rate schedule

#### Fix 5: Evaluation Timing
**Problem**: Validation triggered at wrong iterations
**Solution**: Correct iteration calculation: `epoch * iters_per_epoch`
**Impact**: Proper validation frequency

#### Fix 6: CLAHE Grayscale
**Problem**: Grayscale images stored as 3-channel BGR (all channels identical)
**Solution**: Detect and process as single channel
**Impact**: 3x faster CLAHE processing

### Training Monitoring

**Losses Tracked**:
- `total_loss`: Combined loss
- `loss_cls`: Classification loss (binary: tumor vs background)
- `loss_box_reg`: Bounding box regression loss
- `loss_mask`: Mask segmentation loss
- `loss_rpn_cls`: RPN classification loss
- `loss_rpn_loc`: RPN localization loss

**Validation Metrics** (every epoch):
- Detection mAP@0.5 (main metric)
- Detection mAP@[0.5:0.95]
- Segmentation mAP@[0.5:0.95]
- Logged every 20 iterations
- Full COCO evaluation every 1 epoch

### Training Results (from outputs)

**Epoch 10**:
- Detection mAP@0.5: 34.56%
- Segmentation mAP: 17.96%

**Epoch 25**:
- Detection mAP@0.5: 47.46% 🏆
- Segmentation mAP: 24.97% 🏆

**Best Model Saved**: Model with highest validation mAP@0.5

### Custom Training Loop Features

1. **Custom Evaluator**: Runs COCO evaluation during training
2. **Best Model Tracking**: Saves checkpoint when validation improves
3. **Validation Logging**: Saves results to separate directories
4. **Learning Rate Monitoring**: Logs current LR each validation

### Tools & Libraries Used
- **Detectron2**: Core framework
- **PyTorch**: Deep learning backend
- **CUDA/cuDNN**: GPU acceleration
- **pycocotools**: Evaluation metrics
- **OpenCV (cv2)**: CLAHE preprocessing
- **tensorboard**: Training visualization (implicit)

---

## 📈 STAGE 3: Size-Stratified Sensitivity Analysis

### Purpose
Evaluate detection performance separately for small, medium, and large tumors to understand model limitations.

### Size Categories

**Definition by Bounding Box Area**:
```python
SMALL:  < 96² = 9,216 pixels²
MEDIUM: 96² to 224² = 9,216 to 50,176 pixels²
LARGE:  > 224² = 50,176 pixels²
```

**Medical Justification**:
- Small tumors harder to detect (early stage)
- Large tumors easier but may have different appearance
- Clinical relevance: Early detection crucial

### Evaluation Methodology

#### 1. **Ground Truth Preparation**
- Load COCO annotations from validation set
- Compute bounding box area for each annotation
- Classify into size categories
- Group by image ID

#### 2. **Detection Matching** (IoU-based)
```python
For each image:
  For each ground truth tumor:
    1. Find best matching prediction (highest IoU)
    2. If IoU ≥ 0.5 AND score ≥ 0.15:
        → TRUE POSITIVE
    3. Else:
        → FALSE NEGATIVE (missed tumor)
  
  For each unmatched prediction:
    → FALSE POSITIVE
```

#### 3. **Mask Quality Analysis**

For each matched detection (TP):
```python
mask_iou = intersection(pred_mask, gt_mask) / union(pred_mask, gt_mask)
```

Aggregated by size:
- Mean Mask IoU
- Median Mask IoU
- Standard Deviation

### Results Generated

**Metrics Computed**:
1. **Sensitivity** (Recall): TP / (TP + FN)
2. **True Positives**: Count of correctly detected tumors
3. **False Negatives**: Count of missed tumors
4. **Mask Quality**: Mean/Median/Std IoU of segmentation masks

**Example Results Structure**:
```json
{
  "small": {
    "sensitivity": 0.XXX,
    "tp": N,
    "fn": M,
    "total": N+M,
    "mean_mask_iou": 0.XXX,
    "median_mask_iou": 0.XXX,
    "std_mask_iou": 0.XXX,
    "num_masks": K
  },
  "medium": {...},
  "large": {...}
}
```

### Visualizations Created

1. **Sensitivity Bar Chart**: Detection rate by size
2. **Mask IoU Bar Chart**: Segmentation quality by size
3. **Combined Metrics Plot**: Side-by-side comparison
4. **Detailed Analysis**: Statistical breakdowns

**Publication Quality**:
- 300 DPI resolution
- Professional styling with Seaborn
- Clear labels and legends
- Statistical annotations

### Why This Analysis Matters

1. **Clinical Validation**: Shows if model works across tumor sizes
2. **Bias Detection**: Reveals if model favors large tumors
3. **Performance Transparency**: Honest reporting of limitations
4. **Regulatory Requirement**: FDA/CE marking needs stratified metrics

### Tools Used
- **pycocotools**: COCO format handling
- **numpy**: IoU computations
- **matplotlib**: Plotting
- **seaborn**: Statistical visualization
- **json**: Results export

---

## 🎯 STAGE 4: Image-Level Confusion Matrix

### Purpose
Generate traditional confusion matrix treating detection as **image-level binary classification**.

### Classification Logic

**Ground Truth Labels**:
```python
Positive (Tumor): Image has ≥1 annotation
Negative (Normal): Image has 0 annotations
```

**Prediction Labels**:
```python
Positive: max(detection_scores) > threshold
Negative: max(detection_scores) ≤ threshold OR no detections
```

**Threshold Used**: 0.15 (optimized during validation)

### Confusion Matrix Components

```
                Predicted
              Tumor  Normal
Actual Tumor    TP     FN
       Normal   FP     TN
```

**Definitions**:
- **TP** (True Positive): Tumor image correctly detected
- **TN** (True Negative): Normal image correctly identified
- **FP** (False Positive): Normal image wrongly flagged as tumor
- **FN** (False Negative): Tumor image missed

### Metrics Calculated

#### Primary Metrics
```python
Accuracy  = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)  # PPV
Recall    = TP / (TP + FN)  # Sensitivity
F1-Score  = 2 × (Precision × Recall) / (Precision + Recall)
```

#### Secondary Metrics
```python
Specificity = TN / (TN + FP)  # True Negative Rate
NPV         = TN / (TN + FN)  # Negative Predictive Value
```

### Image-Level vs Instance-Level

**This Stage**: Image-level (binary classification)
- Treats entire image as single prediction
- Uses MAX score aggregation
- Standard for CAD systems

**Later Stages**: Instance-level (object detection)
- Matches individual bounding boxes
- More fine-grained analysis
- Standard for research papers

### Visualization

**Confusion Matrix Heatmap**:
- 2×2 grid with counts
- Color-coded intensity
- Percentage annotations
- Publication-ready styling

### Tools Used
- **sklearn.metrics**: confusion_matrix function
- **matplotlib**: Heatmap creation
- **seaborn**: Professional styling
- **numpy**: Metric calculations

---

## 📊 STAGE 5: Comprehensive AP Metrics Evaluation

### Purpose
Extract **ALL** COCO evaluation metrics from the model and create publication-ready comparison tables.

### COCO Metrics Explained

#### Detection Metrics (Bounding Box)

**AP** (Average Precision):
- Primary COCO metric
- Area under Precision-Recall curve
- Computed at IoU thresholds [0.5:0.95] with step 0.05

**Standard COCO Metrics**:
```
AP       = mAP@[0.5:0.95]  (average over 10 IoU thresholds)
AP@0.5   = mAP@0.5         (loose localization)
AP@0.75  = mAP@0.75        (strict localization)

Size-Stratified:
APs   = AP for small objects  (area < 32²)
APm   = AP for medium objects (32² < area < 96²)
APl   = AP for large objects  (area > 96²)
```

**AR** (Average Recall):
- Maximum recall at different IoU thresholds
- Computed for different max detection limits (1, 10, 100)

#### Segmentation Metrics (Masks)

**Same structure as detection**, but using mask IoU instead of box IoU:
```
Segm AP       = Mask-based mAP@[0.5:0.95]
Segm AP@0.5   = Mask-based mAP@0.5
Segm AP@0.75  = Mask-based mAP@0.75
Segm APs/m/l  = Mask-based size-stratified AP
```

### Evaluation Process

1. **Load Model**: Use trained Mask R-CNN
2. **Register Dataset**: Detectron2 dataset catalog
3. **Run COCO Evaluation**: 
   - COCOeval for bounding boxes
   - COCOeval for segmentation masks
4. **Extract Metrics**: Parse COCO evaluation output
5. **Create Tables**: Format for publication

### Output Files

**CSV Export**:
```csv
Metric Type, AP, AP@0.5, AP@0.75, APs, APm, APl
Detection, 0.XXX, 0.XXX, 0.XXX, 0.XXX, 0.XXX, 0.XXX
Segmentation, 0.XXX, 0.XXX, 0.XXX, 0.XXX, 0.XXX, 0.XXX
```

**Visual Table**:
- Side-by-side comparison
- Detection vs Segmentation
- Color-coded performance
- Publication-ready PNG (300 DPI)

### Why AP Metrics Matter

1. **Standard Benchmark**: Comparable with other papers
2. **Comprehensive**: Single metric captures precision-recall tradeoff
3. **Threshold-Independent**: Doesn't depend on single threshold
4. **Widely Accepted**: Required by most computer vision journals

### Tools Used
- **Detectron2 COCOEvaluator**: Metric computation
- **pandas**: Table creation
- **matplotlib**: Table visualization
- **pycocotools**: COCO format handling

---

## 🔬 STAGE 6: Segmentation Metrics Evaluation

### Purpose
Compute **instance-level** segmentation quality metrics beyond standard COCO metrics.

### Metrics Computed

#### 1. **Mask IoU** (Primary)
```python
IoU = intersection(pred_mask, gt_mask) / union(pred_mask, gt_mask)
```

**Interpretation**:
- 1.0 = Perfect overlap
- 0.5 = 50% overlap (typical threshold)
- 0.0 = No overlap

#### 2. **Dice Coefficient**
```python
Dice = 2 × intersection(pred_mask, gt_mask) / (|pred_mask| + |gt_mask|)
```

**Relationship to IoU**:
```
Dice = 2 × IoU / (1 + IoU)
```

**Why Dice?**
- More forgiving than IoU
- Standard in medical imaging
- Emphasizes overlap vs. union

#### 3. **Boundary F1-Score**

**Purpose**: Measure contour accuracy (edge detection quality)

**Method**:
1. Extract contours from pred_mask and gt_mask
2. Compute distance transform
3. Threshold at 2 pixels
4. Compute precision and recall for boundary pixels
5. F1 = 2PR/(P+R)

**Clinical Relevance**: 
- Sharp boundaries important for surgical planning
- Fuzzy boundaries may indicate uncertainty

#### 4. **Mask Precision & Recall**
```python
Precision = true_positive_pixels / predicted_positive_pixels
Recall    = true_positive_pixels / ground_truth_positive_pixels
```

### Evaluation Levels

**Image-Level** (from earlier stage):
- Binary: Does image have tumor?
- Confusion matrix: TP, TN, FP, FN
- Metrics: Accuracy, Precision, Recall, F1

**Instance-Level** (this stage):
- Per tumor region
- Mask quality for each detection
- Aggregated statistics

### Matching Strategy

```python
For each ground truth mask:
  1. Find prediction with highest box IoU
  2. If IoU ≥ 0.5 and score ≥ 0.15:
     - Compute mask metrics
     - Record as TP
  3. Else:
     - Record as FN
```

### Results Structure

**Detection-Level** (Image):
```json
{
  "accuracy": 0.8209,
  "precision": 0.7918,
  "recall": 0.8776,
  "f1_score": 0.8325,
  "tp": 251, "tn": 212, "fp": 66, "fn": 35
}
```

**Segmentation-Level** (Instance):
```json
{
  "precision": 0.4393,
  "recall": 0.7092,
  "f1_score": 0.5426,
  "avg_iou": 0.7659,
  "avg_dice": 0.8631,
  "boundary_f1": 0.XXX
}
```

### Visualizations

1. **Confusion Matrix**: Image-level classification
2. **Detection Metrics Bar Chart**: Accuracy, Precision, Recall, F1
3. **Segmentation Metrics Bar Chart**: IoU, Dice, Boundary F1
4. **Instance Confusion**: TP/FP/FN counts

### Tools Used
- **OpenCV (cv2)**: Contour extraction
- **scipy.ndimage**: Distance transforms
- **numpy**: Pixel-level computations
- **matplotlib/seaborn**: Visualization
- **json**: Results export

---

## 🎓 STAGE 7: Image-Level Evaluation (Max Aggregation)

### Purpose
Create traditional **binary classification** curves using maximum detection score per image.

### Max Aggregation Method

**Standard Approach** in Computer-Aided Detection:

```python
For each image:
  if len(detections) == 0:
    image_score = 0.0
  else:
    image_score = max(detection_scores)
  
  image_label = 1 if has_tumor_annotation else 0
```

**Why Max?**
- Most confident detection represents image
- Standard for image-level CAD systems
- Converts object detection → image classification
- Enables ROC/PR curve analysis

### Metrics & Visualizations

#### 1. **Precision-Recall Curve**
```
X-axis: Recall (Sensitivity)
Y-axis: Precision (Positive Predictive Value)
Metric: PR-AUC (Area Under Curve)
```

**Interpretation**:
- Higher AUC = Better overall performance
- Curve shape shows precision-recall tradeoff
- Useful when classes imbalanced (tumor vs normal)

#### 2. **ROC Curve**
```
X-axis: False Positive Rate (1 - Specificity)
Y-axis: True Positive Rate (Recall)
Metric: ROC-AUC
```

**Interpretation**:
- Diagonal line = Random classifier (AUC=0.5)
- Perfect classifier: AUC=1.0
- More robust to class imbalance than accuracy

#### 3. **F1-Score vs Threshold**
```
X-axis: Classification threshold
Y-axis: F1-Score
```

**Purpose**: Find optimal operating point

**Optimal Threshold**:
- Maximizes F1-Score
- Balances precision and recall
- Used for deployment

#### 4. **Confusion Matrix** (at optimal threshold)
- 2×2 grid
- Counts at best F1 threshold
- Shows practical performance

### Output Files

**Plots**:
- `pr_curve.png`
- `roc_curve.png`
- `f1_vs_threshold.png`
- `confusion_matrix.png`

**JSON Results**:
```json
{
  "pr_auc": 0.XXX,
  "roc_auc": 0.XXX,
  "optimal_threshold": 0.XXX,
  "best_f1": 0.XXX,
  "precision_at_optimal": 0.XXX,
  "recall_at_optimal": 0.XXX
}
```

### Clinical Decision Support

**Threshold Selection**:
- **High threshold** (e.g., 0.5): Fewer false alarms, may miss tumors
- **Low threshold** (e.g., 0.1): Catch more tumors, more false alarms
- **Optimal** (max F1): Balance for clinical workflow

**In Practice**:
```
Screening:     Low threshold (high sensitivity)
Confirmation:  High threshold (high precision)
Research:      Optimal threshold (balanced)
```

### Tools Used
- **sklearn.metrics**: precision_recall_curve, roc_curve, auc
- **numpy**: Threshold optimization
- **matplotlib**: Curve plotting

---

## 🔬 STAGE 8: Detection-Level Evaluation (Journal Standard)

### Purpose
Compute metrics at the **detection level** (bounding box level) - the standard for research papers.

### Detection-Level vs Image-Level

**Image-Level** (previous stage):
- One score per image (max aggregation)
- Binary classification: tumor/normal
- Suitable for CAD systems

**Detection-Level** (this stage):
- One score per bounding box
- Matches boxes using IoU
- Suitable for research papers
- More fine-grained analysis

### Box Matching Algorithm

```python
For each ground truth box:
  1. Compute IoU with all predicted boxes
  2. Find prediction with highest IoU
  3. If IoU ≥ 0.5 AND score ≥ threshold:
     - Mark as TRUE POSITIVE (matched)
     - Remove from candidates
  4. Else:
     - Mark ground truth as FALSE NEGATIVE (missed)

For each unmatched prediction (IoU < 0.5 or extra):
  - Mark as FALSE POSITIVE (false alarm)
```

**IoU Threshold**: 0.5 (COCO standard)

### Confusion Matrix (Detection-Level)

```
Not a 2×2 matrix (no True Negatives in object detection)

TP: Correctly detected boxes (IoU ≥ 0.5)
FP: Incorrect detections (IoU < 0.5 or duplicate)
FN: Missed ground truth boxes
```

### Metrics at Detection Level

**At Fixed Threshold** (0.15):
```python
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1-Score  = 2 × (Precision × Recall) / (Precision + Recall)
```

**Threshold-Independent**:
```python
PR-AUC: Area under Precision-Recall curve
AP: Average Precision (same as PR-AUC)
ROC-AUC: Area under ROC curve
```

### Visualizations (Individual Figures)

**6 Separate Publication-Quality Plots**:

1. **Precision-Recall Curve**
   - Shows tradeoff across all thresholds
   - Includes PR-AUC value
   - Shaded area under curve

2. **ROC Curve**
   - TPR vs FPR
   - Includes ROC-AUC value
   - Diagonal reference line

3. **F1-Score vs Threshold**
   - Shows optimal operating point
   - Marks maximum F1
   - Helps choose deployment threshold

4. **Confusion Matrix**
   - 3-value (TP, FP, FN)
   - At optimal threshold
   - Color-coded heatmap

5. **Metrics vs Threshold**
   - Precision, Recall, F1 on same plot
   - Shows how metrics change with threshold
   - Helps understand tradeoffs

6. **Metrics Summary Table**
   - All key numbers in table format
   - Publication-ready
   - Includes both fixed-threshold and curve metrics

### JSON Export

**Comprehensive Results**:
```json
{
  "threshold": 0.15,
  "total_gt": 337,
  "total_detections": 542,
  "tp": 239,
  "fp": 303,
  "fn": 98,
  "precision": 0.4409,
  "recall": 0.7093,
  "f1_score": 0.5430,
  "pr_auc": 0.5652,
  "average_precision": 0.5652,
  "roc_auc": 0.8346,
  "optimal_threshold": 0.1234,
  "best_f1": 0.5623
}
```

### Why Detection-Level Metrics?

1. **Research Standard**: What peer-reviewed papers use
2. **Fine-Grained**: Shows per-box performance
3. **Localization Quality**: Incorporates IoU threshold
4. **Comparable**: Standard across object detection papers

### Differences from Image-Level

| Aspect | Image-Level | Detection-Level |
|--------|-------------|-----------------|
| Granularity | Whole image | Each box |
| Aggregation | Max score | Box matching |
| TN exists? | Yes | No |
| Clinical use | CAD systems | Research |
| Sensitivity | Higher | Lower (stricter) |

### Tools Used
- **numpy**: Box matching, IoU computation
- **sklearn.metrics**: Curve computation
- **matplotlib**: Multi-panel plotting
- **json**: Results export

---

## 🛠️ Complete Technology Stack

### Core Frameworks
```
PyTorch 1.13+          - Deep learning backend
Detectron2 0.6+        - Object detection framework
CUDA 11.x              - GPU acceleration
cuDNN 8.x              - Neural network primitives
```

### Computer Vision
```
OpenCV (cv2) 4.x       - Image processing, CLAHE
pycocotools 2.x        - COCO format, mask encoding
Pillow (PIL)           - Image I/O
albumentations         - Data augmentation
```

### Data Science
```
NumPy 1.21+            - Numerical computing
pandas 1.3+            - Tabular data handling
scikit-learn 1.0+      - Metrics, train/test split
SciPy 1.7+             - Distance transforms
```

### Visualization
```
matplotlib 3.5+        - Plotting
seaborn 0.11+          - Statistical visualization
tqdm 4.x               - Progress bars
```

### Utilities
```
json                   - Data serialization
pathlib                - Path handling
collections            - Data structures
```

### Hardware
```
GPU: NVIDIA Tesla T4 (16GB VRAM)
CPU: Multi-core (for data loading)
RAM: 32GB+ (recommended)
Storage: SSD (fast data access)
```

---

## 📊 Key Results Summary (from Outputs)

### Training Performance

**Validation Set** (580 images):
- Detection mAP@0.5: **47.46%** (epoch 25)
- Detection mAP@[0.5:0.95]: **19.96%**
- Segmentation mAP: **24.97%**

### Test Set Performance

**Image-Level Classification**:
```
Accuracy:  82.09%
Precision: 79.18%
Recall:    87.76%
F1-Score:  83.25%
```

**Instance-Level Segmentation**:
```
Mask Precision: 43.93%
Mask Recall:    70.92%
Mask F1:        54.26%
Average IoU:    76.59%
Average Dice:   86.31%
```

**Detection-Level** (Box Matching):
```
Precision:     44.09%
Recall:        70.93%
F1-Score:      54.30%
PR-AUC:        56.52%
ROC-AUC:       83.46%
```

---

## 🎯 Critical Design Decisions & Justifications

### 1. Patient-Level Splitting
**Why**: Prevents data leakage when same patient has multiple images
**Impact**: More realistic generalization performance
**Tradeoff**: Slightly reduced training set diversity

### 2. Center-Aware Stratification
**Why**: Controls for institution-specific biases
**Impact**: Model generalizes across centers
**Tradeoff**: More complex splitting logic

### 3. Single "Tumor" Class
**Why**: Simplifies detection task
**Impact**: Higher recall, sufficient for screening
**Tradeoff**: Loses benign/malignant distinction (separate classifier needed)

### 4. IoU Threshold = 0.5
**Why**: COCO standard, widely accepted
**Impact**: Balanced localization requirement
**Tradeoff**: May accept somewhat loose boxes

### 5. Score Threshold = 0.15
**Why**: Optimized for F1-Score on validation set
**Impact**: Balances precision and recall
**Tradeoff**: May need adjustment for specific clinical use case

### 6. CLAHE Preprocessing
**Why**: Enhances contrast in X-rays
**Impact**: Better feature extraction
**Tradeoff**: Slight computational overhead

### 7. Mixed Precision Training
**Why**: Faster training, lower memory
**Impact**: 2x speedup on T4 GPU
**Tradeoff**: Requires careful hyperparameter tuning

### 8. Multiple Evaluation Levels
**Why**: Comprehensive performance analysis
**Impact**: Satisfies both clinical (CAD) and research needs
**Tradeoff**: More complex codebase

---

## 📖 Publication-Ready Outputs

### Figures Generated (All 300 DPI)

**Size Analysis**:
- Sensitivity by tumor size
- Mask IoU by tumor size
- Combined metrics comparison

**Confusion Matrices**:
- Image-level (2×2)
- Detection-level (TP/FP/FN)
- Instance-level breakdown

**Performance Curves**:
- Precision-Recall curve + AUC
- ROC curve + AUC
- F1-Score vs threshold
- Metrics vs threshold

**Comparison Tables**:
- Detection vs Segmentation AP
- Metrics summary table

### Numerical Results

**JSON Files**:
- Detection metrics summary
- Segmentation metrics summary
- Size-stratified results
- Per-image predictions
- Complete evaluation summary

**CSV Files**:
- COCO metrics export
- Metadata with predictions

---

## 🚨 Limitations & Caveats

### Documented in Code

1. **Approximated Patient IDs**
   - No explicit patient identifiers in dataset
   - Rare edge cases may have ambiguity
   - Must be disclosed in Methods section

2. **Center Imbalance**
   - Center 1 dominates (69% of patients)
   - Model may favor Center 1 characteristics
   - Mitigated by stratified splitting

3. **Class Imbalance**
   - Malignant tumors underrepresented (11.4%)
   - Normal images numerous (21.6%)
   - Benign dominant (67.0%)

4. **Single Category Detection**
   - Benign/Malignant merged
   - Separate classification needed for diagnosis
   - Detection is first stage only

5. **Dataset Size**
   - 1,008 patients total
   - 152 patients in test set
   - May limit generalization claims

---

## ✅ Best Practices Implemented

### Data Science
✅ Patient-level splitting (no leakage)
✅ Stratified splitting (balanced distributions)
✅ Label-free patient fingerprinting
✅ Validation set for hyperparameter tuning
✅ Test set held out for final evaluation

### Machine Learning
✅ Mixed precision training (efficiency)
✅ Gradient clipping (stability)
✅ Learning rate warmup (training stability)
✅ Checkpoint saving (reproducibility)
✅ Multiple evaluation metrics (comprehensive)

### Software Engineering
✅ Modular code structure
✅ Extensive documentation
✅ Progress bars (user experience)
✅ Error handling
✅ Configurable parameters

### Research Integrity
✅ All limitations documented
✅ Methodological choices justified
✅ Multiple evaluation perspectives
✅ Publication-ready figures
✅ Reproducible pipeline

---

## 🎓 Learning Points

### For Medical AI Researchers

1. **Data Leakage**: Always split at patient level, not image level
2. **Bias Control**: Stratify by institution/center when multi-site
3. **Evaluation Levels**: Report both image-level (clinical) and instance-level (research) metrics
4. **Threshold Selection**: Optimize on validation set, report on test set
5. **Size Stratification**: Analyze performance across tumor sizes
6. **COCO Format**: Standard for detection, well-supported

### For Deep Learning Practitioners

1. **Detectron2**: Production-ready framework, beats custom implementations
2. **Mixed Precision**: Significant speedup on modern GPUs (T4, V100, A100)
3. **CLAHE**: Essential preprocessing for medical X-rays
4. **Checkpoint Management**: Save best model during training
5. **Bug Fixes**: Document and test all critical fixes
6. **Validation Frequency**: Balance between training time and monitoring

### For Medical Imaging

1. **Instance Segmentation**: Mask R-CNN provides pixel-level delineation
2. **Multi-Stage Evaluation**: Detection → Localization → Segmentation quality
3. **Clinical Metrics**: Sensitivity critical for screening
4. **Boundary Analysis**: Edge quality matters for surgical planning
5. **False Positives**: Must be minimized in clinical deployment

---

## 📚 References & Citations

### Models
- **Mask R-CNN**: He et al., "Mask R-CNN", ICCV 2017
- **ResNet**: He et al., "Deep Residual Learning", CVPR 2016
- **FPN**: Lin et al., "Feature Pyramid Networks", CVPR 2017

### Frameworks
- **Detectron2**: Wu et al., Facebook AI Research
- **PyTorch**: Paszke et al., "PyTorch: An Imperative Style", NeurIPS 2019

### Metrics
- **COCO Metrics**: Lin et al., "Microsoft COCO", ECCV 2014
- **IoU**: Jaccard Index
- **Dice**: Sørensen–Dice coefficient

### Preprocessing
- **CLAHE**: Zuiderveld, "Contrast Limited AHE", Graphics Gems IV

---

## 🎯 Conclusion

This notebook implements a **complete, production-ready pipeline** for bone tumor detection using state-of-the-art deep learning. It demonstrates:

1. **Scientific Rigor**: Patient-level splitting, bias control, comprehensive evaluation
2. **Technical Excellence**: Modern architecture, efficient training, robust evaluation
3. **Clinical Relevance**: Multiple evaluation levels, size stratification, threshold optimization
4. **Publication Quality**: Professional figures, complete metrics, documented limitations
5. **Reproducibility**: Clear code, saved outputs, version-controlled

The pipeline is suitable for:
- **Research Papers**: All standard metrics reported
- **Clinical Validation**: CAD-style image-level evaluation
- **Regulatory Submission**: Comprehensive safety/efficacy data
- **Further Development**: Modular, well-documented code

**Total Lines of Code**: ~6,800 (across 8 major cells)
**Execution Time**: ~4-6 hours (training) + ~30 minutes (evaluation)
**Key Innovation**: Combines patient-level data splitting with comprehensive multi-level evaluation
