# 🦴 BoneFusionNet

**BoneFusionNet** is a detection-guided multimodal deep learning framework for bone tumor classification using radiographic images and clinical metadata.

It integrates tumor detection, ROI-based classification, and structured clinical features to provide accurate, reliable, and clinically meaningful predictions.

---

## 🚀 Features

- 🔍 **Tumor Detection** using Faster R-CNN  
- 🧠 **ROI-based Classification** using ConvNeXt  
- 🔗 **Multimodal Fusion (FusionNet)** combining image + clinical data  
- 📊 **Calibrated Predictions** for reliable decision-making  
- 🔬 **Explainability** with Grad-CAM++ and Integrated Gradients  
- 🌐 **Web Interface** using Streamlit  

---

## 🧩 Project Pipeline

1. **Detection:** Localize tumor regions from X-ray images  
2. **ROI Extraction:** Crop detected regions  
3. **Classification:** Predict benign vs malignant  
4. **Fusion:** Combine image features with clinical metadata  
5. **Output:** Final prediction with explainability  

---

## 📊 Results

- ✅ **ROC-AUC:** 0.903  
- ✅ **PR-AUC:** 0.757  
- ✅ Improved performance over image-only and clinical-only models  
- ✅ Reduced false positives with better balance between sensitivity and specificity  

---
