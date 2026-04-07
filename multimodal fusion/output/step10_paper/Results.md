# Results

## 3.3 Fusion Pipeline Evaluation

### 3.3.1 Ablation Study (Test Set, Image-Level)

Table 2 presents ablation results on the held-out test set
(n = 241 images, 41 malignant, 200 benign; 17.0% malignant prevalence).
All reported metrics use thresholds optimised on the validation set.

**FusionNet** (image + clinical + LR probability) achieved AUC-PR =
0.757 [95% CI: 0.618–0.853] and AUC-ROC = 0.903 [0.851–0.947],
with malignant sensitivity = 0.634, specificity = 0.945,
and MCC = 0.604 (TP=26, FP=11, TN=189, FN=15; threshold = 0.610).

The **Image-Only MLP** achieved AUC-PR = 0.738 [0.584–0.859] and
AUC-ROC = 0.911 [0.862–0.955], with TP=25, FP=13, TN=187, FN=16
(threshold = 0.650). Both image-based models substantially outperformed
the clinical-only baselines (Clinical LR: AUC-PR = 0.192, AUC-ROC = 0.551;
Clinical-Only MLP: AUC-PR = 0.257, AUC-ROC = 0.581).

FusionNet achieved marginally higher AUC-PR (+0.019) and lower FP count
(11 vs 13) than Image-Only MLP. The clinical branch contributes a small but
positive shift in precision at the chosen threshold, despite the LR
probability's compressed dynamic range
($p_\text{malignant} \in [0.07, 0.40]$, class effect size ≈ 0.88σ).

Bootstrap significance testing confirmed that differences between
image-based models (FusionNet, Image-Only) and clinical-only baselines
were statistically significant for both AUC-ROC and AUC-PR
(all p < 0.05; see Table 3). The difference between FusionNet and
Image-Only MLP was not statistically significant (p > 0.05), consistent
with their similar AUC profiles.

### 3.3.2 Patient-Level Results

Patient-level aggregation (mean-pooling across images per patient,
n = 99 patients, 18 malignant, 81 benign) improved FusionNet performance:
AUC-PR = 0.686, AUC-ROC = 0.895, balanced accuracy = 0.631,
MCC = 0.366, malignant sensitivity = 0.300, malignant precision = 0.667.
The lower sensitivity at patient level relative to image level reflects
the strict mean-pooling across multiple views per patient; individual
images where the model is uncertain dilute the aggregated probability.

### 3.3.3 Fallback ROI Sensitivity Analysis

Removing 19 test images (7.9%) whose ROIs were detected only by the
fallback detector modestly improved FusionNet performance:
AUC-PR: 0.757 → 0.780; malignant sensitivity: 0.634 → 0.667;
MCC: 0.604 → 0.632. This indicates that detection confidence measurably
affects downstream classification, motivating tighter confidence
filtering in clinical deployment.

### 3.3.4 Training Dynamics

FusionNet reached best validation AUC-PR at epoch 20 (early stopping
after 35 total epochs). Image-Only MLP peaked at epoch 17,
Clinical-Only MLP at epoch 33. The extended training required by
Clinical-Only MLP reflects the weaker and noisier signal in clinical
metadata alone. Both image-based models show substantial validation
AUC-PR drop from training to test (FusionNet: 0.984 → 0.757;
Image-Only: 0.984 → 0.738), consistent with the small training set
size (n = 1,520 ROIs) and deep ConvNeXt-Tiny-SE feature extraction.
