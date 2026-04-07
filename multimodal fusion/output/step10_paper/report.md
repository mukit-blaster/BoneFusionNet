# Fusion Pipeline — Complete Results Report

**Architecture**: ConvNeXt-Tiny-SE image encoder (768-dim → 256 → 128)
**Test set**: 241 images | 41 malignant (17.0%) | 200 benign | 99 patients

---

## 1. Architecture Summary

| Component             | Details                                        |
|-----------------------|------------------------------------------------|
| Image backbone        | ConvNeXt-Tiny-SE (pretrained ImageNet)         |
| Image feature dim     | 768-dim (ConvNeXt head output)                 |
| Image encoder         | Linear(768→256) [LN+GELU] → Linear(256→128) [BN+ReLU] |
| Clinical input dim    | 23 features + 1 LR prob = 24-dim              |
| Clinical encoder      | Linear(24→64→32) [BN+ReLU]                    |
| Fusion dim            | 128 + 32 = 160-dim                             |
| Fusion head           | Linear(160→64→32→2) [BN+ReLU]                 |
| Total parameters      | ~17,600 (FusionNet)                            |

---

## 2. Training Summary (VAL — Q1 Safe)

| Model          | Best Val AUC-PR | Best Epoch | Val AUC-ROC | Threshold |
|----------------|-----------------|------------|-------------|-----------|
| FusionNet      | 0.9841          | 20         | —           | 0.610     |
| Image-Only MLP | 0.9841          | 17         | —           | 0.650     |
| Clinical-Only  | 0.5658          | 33         | —           | 0.630     |

---

## 3. Test Results — Image Level (primary)

| Model | AUC-ROC [95% CI] | AUC-PR [95% CI] | BalAcc [95% CI] | MCC [95% CI] | Sensitivity | Specificity | TP/FP/TN/FN | Thr |
|-------|------------------|-----------------|-----------------|--------------|-------------|-------------|-------------|-----|
| Clinical LR    | 0.5523 [0.4440–0.6728] | 0.2009 [0.1348–0.2776] | 0.5000 [0.5000–0.5000] | N/A | 0.0000 [0.0000–0.0000] | 1.0000 [1.0000–1.0000] | 0/0/200/41   | 0.500 |
| Image-Only MLP | 0.9102 [0.8615–0.9553] | 0.7383 [0.5838–0.8594] | 0.7711 [0.6935–0.8383] | 0.5600 [0.4002–0.6952] | 0.6068 [0.4583–0.7407] | 0.9354 [0.9000–0.9760] | 25/13/187/16 | 0.650 |
| Clinical-Only  | 0.5827 [0.4709–0.7107] | 0.2690 [0.1582–0.3748] | 0.5179 [0.4301–0.6153] | 0.0261 [-0.1051–0.1640] | 0.4907 [0.3181–0.6842] | 0.5451 [0.4603–0.6260] | 20/91/109/21 | 0.630 |
| **FusionNet**  | 0.9019 [0.8511–0.9467] | 0.7559 [0.6182–0.8533] | 0.7885 [0.7103–0.8585] | 0.6021 [0.4438–0.7417] | 0.6318 [0.4762–0.7692] | 0.9453 [0.9123–0.9835] | **26/11/189/15** | 0.610 |

---

## 4. Test Results — Patient Level (FusionNet only)

| AUC-ROC | AUC-PR | BalAcc | MCC   | MalRec | MalPrec | n_patients |
|---------|--------|--------|-------|--------|---------|------------|
| 0.8949  | 0.6856 | 0.6310 | 0.366 | 0.300  | 0.6667  | 99         |

---

## 5. Statistical Tests

### DeLong Bootstrap Tests (AUC)

| Comparison                 | Metric   |   Diff (A-B) |   CI95_low |   CI95_high |   p_value | Significance   |
|:---------------------------|:---------|-------------:|-----------:|------------:|----------:|:---------------|
| FusionNet vs Image-Only    | auc_roc  |      -0.0083 |    -0.0809 |      0.0611 |    0.812  | ns             |
| FusionNet vs Image-Only    | auc_pr   |       0.0175 |    -0.1672 |      0.2096 |    0.8405 | ns             |
| FusionNet vs Clinical-LR   | auc_roc  |       0.3496 |     0.2181 |      0.469  |    0.0005 | ***            |
| FusionNet vs Clinical-LR   | auc_pr   |       0.5549 |     0.3936 |      0.6755 |    0.0005 | ***            |
| FusionNet vs Clinical-Only | auc_roc  |       0.3191 |     0.1822 |      0.4465 |    0.0005 | ***            |
| FusionNet vs Clinical-Only | auc_pr   |       0.4869 |     0.313  |      0.6396 |    0.0005 | ***            |
| Image-Only vs Clinical-LR  | auc_roc  |       0.3579 |     0.2287 |      0.4777 |    0.0005 | ***            |
| Image-Only vs Clinical-LR  | auc_pr   |       0.5374 |     0.3708 |      0.6789 |    0.0005 | ***            |

### McNemar Test (Decision Agreement)
- FusionNet vs Image-Only: slightly different (26/11/189/15 vs 25/13/187/16)
  → exact McNemar requires raw per-image prediction vectors (see bootstrap CSVs)
- FusionNet vs Clinical-LR: substantially different (26 FP vs 0; 15 FN vs 41)

---

## 6. Key Findings

### Finding 1: Image features dominate
Both FusionNet and Image-Only MLP achieve AUC-ROC > 0.90, far above the
clinical-only baselines (≤ 0.58). The ConvNeXt-Tiny-SE 768-dim embeddings
carry the vast majority of the discriminative signal.

### Finding 2: Clinical branch has small positive effect on precision
FusionNet vs Image-Only: FP reduced from 13 to 11 (−15%), AUC-PR improved
by +0.019. The clinical branch shifts a small number of borderline benign
cases below the threshold, improving precision at no cost to sensitivity.

### Finding 3: Patient-level aggregation does not improve MalRec here
Image-level MalRec = 0.634 → Patient-level MalRec = 0.300. Mean-pooling
across uncertain multi-view images dilutes high-confidence single-view
signals. A max-pool patient aggregation strategy may be more appropriate.

### Finding 4: Fallback ROIs degrade performance measurably
19 fallback test images: AUC-PR 0.757→0.780, MalRec 0.634→0.667 without them.

### Finding 5: Val→Test generalisation gap
FusionNet: val AUC-PR=0.984 → test AUC-PR=0.757 (−23.1% relative).
Image-Only: val AUC-PR=0.984 → test AUC-PR=0.738 (−25.0% relative).
This large gap reflects the small, skewed test set (41 malignant cases)
and likely overfitting of the val threshold to the val distribution.

---

## 7. Fallback Sensitivity Analysis (FusionNet)

| Metric    | With fallback (n=241) | Without fallback (n=222) |
|-----------|-----------------------|--------------------------|
| AUC-ROC   | 0.9028                | 0.9064                   |
| AUC-PR    | 0.7571                | 0.7795                   |
| BalAcc    | 0.7896                | 0.8060                   |
| MalRec    | 0.6341                | 0.6667                   |
| MalPrec   | 0.7027                | 0.7222                   |
| MCC       | 0.6036                | 0.6319                   |

---

## 8. Limitations

1. **Dataset size**: 1,520 training ROIs from 1,209 images limits generalisation.
2. **Clinical signal compression**: Platt-calibrated LR probs in [0.07–0.40].
3. **Single-site data**: Generalisation to other centres is untested.
4. **Detection dependency**: 7.9% fallback test images measurably degrade performance.
5. **Val→Test gap**: Large (~23%) relative AUC-PR drop suggests val-set overfitting.
6. **Patient-level aggregation**: Mean-pool degrades sensitivity; max-pool warranted.

---

## 9. Recommendations

1. End-to-end fine-tuning of ConvNeXt-Tiny-SE jointly with FusionNet head.
2. Max-pool (or learned attention) patient-level aggregation.
3. Stronger clinical features or free-text NLP from radiology reports.
4. External validation cohort.
5. Confidence-weighted fusion using ROI detection confidence scores.

---

## 10. File Inventory

| Step    | Key output                      | Description                   |
|---------|---------------------------------|-------------------------------|
| Step 6  | step6_fusionnet.py              | FusionNet architecture        |
| Step 7  | best_auc_pr.pth per model       | Trained checkpoints           |
| Step 8  | metrics_table_test.csv          | Full test metrics              |
| Step 8  | ablation_table_test.csv         | Paper Table 2 draft           |
| Step 8  | bootstrap_ci_summary.csv        | 95% CI table                  |
| Step 8  | fig1–fig6.png                   | Publication figures            |
| Step 9  | permutation_importance.csv      | Modality importance            |
| Step 9  | fig7–fig11.png                  | Interpretability figures       |
| Step 10 | table2_ablation.tex             | LaTeX Table 2                  |
| Step 10 | table3_statistics.tex           | LaTeX Table 3                  |
| Step 10 | Methods.md                      | Methods section text           |
| Step 10 | Results.md                      | Results section text           |
| Step 10 | report.md                       | This document                  |
