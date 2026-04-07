# Methods

## 2.5 Multimodal Fusion Pipeline

### 2.5.1 Overview

We developed a multimodal fusion pipeline combining (i) appearance features
extracted by a ConvNeXt-Tiny with Squeeze-and-Excitation (SE) blocks from
detected bone tumour regions of interest (ROIs), (ii) structured clinical
metadata, and (iii) calibrated malignancy probabilities from a logistic
regression (LR) clinical model trained in Stage 2 of this study.

### 2.5.2 Feature Extraction

**Image features.** ConvNeXt-Tiny with Squeeze-and-Excitation (SE) blocks,
pre-trained on ImageNet, was applied to each detected ROI. The
768-dimensional feature vector from the ConvNeXt head (after global average
pooling and layer normalisation) was extracted and saved without dimensionality
reduction. Features were normalised per-dimension using StandardScaler
statistics fit on training ROIs only and applied unchanged to validation and
test sets (Step 3, no-augmentation variant).

**Clinical features.** A total of 23 structured clinical features were used,
covering patient demographics (age, gender), affected bone locations
(hand, ulna, radius, humerus, femur, tibia, fibula, hip bone, foot),
joint involvement (ankle-joint, knee-joint, hip-joint, wrist-joint,
elbow-joint, shoulder-joint), body region (upper limb, lower limb, pelvis),
and radiographic view (frontal, lateral, oblique).

**Clinical LR probability.** A calibrated malignancy probability scalar
$p_\text{malignant} \in [0, 1]$ was extracted from the Stage 2
GroupCalibratedEnsemble (logistic regression + Platt scaling). At the image
level, this produced a single probability per patient–image pair, which was
broadcast to all associated ROIs. The resulting scalar captures the clinical
prior for malignancy based on patient metadata alone.

### 2.5.3 FusionNet Architecture

We designed a dual-encoder intermediate fusion network (FusionNet). The
architecture consists of:

- **Image encoder** (768 → 256 → 128):
  - Stage 1: ConvNeXt-style projection block —
    Linear(768→256) → LayerNorm(256) → GELU → Dropout(0.40).
    LayerNorm and GELU match ConvNeXt's own head style.
  - Stage 2: standard dense block —
    Linear(256→128) → BatchNorm(128) → ReLU → Dropout(0.30).
  - Output: 128-dimensional image embedding.

- **Clinical encoder** (24 → 64 → 32):
  The 23 clinical features and 1 LR probability scalar are concatenated
  (24-dimensional input) and passed through two dense blocks
  (Linear–BatchNorm–ReLU–Dropout): 24→64→32 (dropout = 0.20).
  The LR probability is fed into this branch rather than treated as a
  third independent encoder, allowing the network to learn its relative
  contribution.
  - Output: 32-dimensional clinical embedding.

- **Fusion head** (160 → 64 → 32 → 2):
  The concatenated 160-dimensional embedding (128 image + 32 clinical)
  is passed through two dense blocks (160→64→32, dropout 0.40/0.30)
  followed by a linear classifier (32→2).
  - Output: 2-class logits.

All linear layers use Kaiming uniform initialisation. Bias terms are omitted
from layers followed by batch normalisation. LayerNorm layers use default
initialisation (weight=1, bias=0).

Two ablation variants sharing the same backbone were trained:
**Image-Only MLP** (same 768→256→128 image encoder + smaller head 128→32→2,
no clinical input) and **Clinical-Only MLP** (clinical encoder 24→64→32 +
head 32→16→2, no image input).

### 2.5.4 Training

All models were trained with the following configuration:

| Hyperparameter          | Value                                         |
|-------------------------|-----------------------------------------------|
| Loss function           | Focal Loss (α=0.65, γ=1.2)                   |
| Optimiser               | AdamW (lr=1×10⁻³, wd=1×10⁻³)                |
| LR scheduler            | Linear warmup (5 ep) + ReduceLROnPlateau      |
|                         | (patience=7, factor=0.5, min_lr=1×10⁻⁶)      |
| Gradient clipping       | max-norm = 1.0                                |
| Batch size              | 64                                            |
| Class imbalance         | WeightedRandomSampler (×2.0 malignant weight) |
|                         | + Focal Loss α=0.65                           |
| Early stopping          | patience=15, monitor=val AUC-PR, min_epochs=20|
| Label smoothing (train) | 0.05                                          |
| Random seed             | 42                                            |

### 2.5.5 Threshold Optimisation

Classification thresholds were optimised on the **validation set only**
using a composite score subject to the clinical constraint
malignant recall ≥ 0.85:

$$\text{score} = 1.00 \cdot \text{BalAcc} + 0.25 \cdot \text{MCC}
               + 0.10 \cdot \text{MacroF1}
               - 0.10 \cdot |\hat{r} - r|$$

where $r$ is the true malignant prevalence and $\hat{r}$ is the predicted
positive rate. Optimised thresholds were applied **unchanged** to the test set.

### 2.5.6 Evaluation

**Aggregation.** For images with multiple detected ROIs, softmax probabilities
were averaged across ROIs before applying the threshold (mean-pooling). The
final image-level probability is the mean of all ROI softmax vectors.
Patient-level predictions were computed by averaging image-level probabilities
across all images per patient.

**Metrics.** Primary metrics: AUC-PR and AUC-ROC. Secondary metrics: balanced
accuracy (BalAcc), Matthews correlation coefficient (MCC), malignant
sensitivity (recall), malignant precision, and specificity. All metrics were
computed at image level unless stated otherwise.

**Bootstrap confidence intervals.** 95% CIs were computed by patient-level
resampling ($N_\text{boot}=2{,}000$) to account for within-patient ROI
correlation. Statistical comparisons used bootstrap-based permutation tests
(10,000 permutations).

**Fallback sensitivity analysis.** A total of 19 test images (7.9%) had ROIs
detected only by the fallback low-confidence detector. A sensitivity analysis
compared metrics with and without these images.
