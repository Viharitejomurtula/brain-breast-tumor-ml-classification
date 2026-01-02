# 🧠 Multi-Modal Tumor Classification with Explainable AI

<div align="center">

**Brain + Breast MRI Classification | Custom TrigConv2D Architecture | GRAD-CAM & Integrated Gradients**

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[View Demo Notebook](notebooks/02_explainability_demo.ipynb) • [Explainability Results](#-explainability-deep-dive) • [Contact](#-contact)

</div>

---

## 🎯 Project Summary

> **Not just classification — understanding *why* the model decides what it decides.**

This project builds a **6-class tumor classifier** across two MRI modalities (brain & breast) with a focus on **explainable AI**. The key insight: different attribution methods reveal different aspects of model reasoning.

| What I Built | Why It Matters |
|--------------|----------------|
| Multi-modal CNN classifier | Handles anatomically different scan types in one model |
| Custom `TrigConv2D` layer | Novel feature extraction using sin/cos frequency kernels |
| Dual explainability pipeline | GRAD-CAM + Integrated Gradients reveal complementary insights |
| Low-confidence analysis | Shows model uncertainty correlates with diffuse explanations |

---

## ⚡ Key Results

<table>
<tr>
<td width="50%">

### 📊 Model Performance
| Metric | Value |
|--------|-------|
| **Test Accuracy** | 95.95% |
| **Classes** | 6 |
| **Input Size** | 128 × 128 × 3 |
| **Total Test Samples** | 4,739 |

</td>
<td width="50%">

### 🔬 Explainability Insight
> *"GRAD-CAM captures where the model looks to classify **modality**. Integrated Gradients reveals where the **pathology** is."*

This distinction is critical for clinical interpretability.

</td>
</tr>
</table>

---

## 🔬 Explainability Deep Dive

### The Core Discovery

The model learns a **two-step decision hierarchy**:

```
Step 1: Modality Recognition    →    "Is this a brain or breast scan?"
Step 2: Tumor Classification    →    "What type of tumor (if any)?"
```

**Different explainability methods expose different steps:**

| Method | What It Reveals | Resolution | Best For |
|--------|-----------------|------------|----------|
| **GRAD-CAM** | Broad anatomical attention zones | Coarse (feature map) | Verifying modality focus |
| **Integrated Gradients** | Pixel-level tumor attribution | Fine (input resolution) | Clinical interpretability |

---

### 📸 Case Study Visualizations

#### Case 1: High-Confidence Breast MRI (Benign)
| Original | GRAD-CAM | Integrated Gradients | Prediction |
|----------|----------|---------------------|------------|
| ![Original](assets/case1_original.png) | ![GRAD-CAM](assets/case1_gradcam.png) | ![IG](assets/case1_ig.png) | ✅ Benign (100.0%) |

> **Interpretation:** GRAD-CAM highlights breast tissue boundaries (modality). IG isolates the lesion core (pathology).

---

#### Case 2: Brain MRI (Glioma Tumor)
| Original | GRAD-CAM | Integrated Gradients | Prediction |
|----------|----------|---------------------|------------|
| ![Original](assets/case2_original.png) | ![GRAD-CAM](assets/case2_gradcam.png) | ![IG](assets/case2_ig.png) | ✅ Glioma (100.0%) |

> **Interpretation:** GRAD-CAM spreads across brain structure. IG pinpoints hyperintense tumor regions aligned with radiologist attention.

---

#### Case 3: Human Perception Bias
| Original | GRAD-CAM | Integrated Gradients | Prediction |
|----------|----------|---------------------|------------|
| ![Original](assets/case3_original.png) | ![GRAD-CAM](assets/case3_gradcam.png) | ![IG](assets/case3_ig.png) | ✅ No Tumor (100.0%) |

> **Interpretation:** Even when humans might confuse the scan for another body part, the explainability maps show the model is still reasoning like a “brain detector".
---

#### Case 4: Low Confidence Prediction
| Original | GRAD-CAM | Integrated Gradients | Prediction |
|----------|----------|---------------------|------------|
| ![Original](assets/case3_original.png) | ![GRAD-CAM](assets/case3_gradcam.png) | ![IG](assets/case3_ig.png) | ⚠️ No Tumor (57.4%) |

> **Interpretation:** When confidence drops, both explanations become diffuse. This correlation between uncertainty and unfocused attribution indicates the model isn't hallucinating — it's appropriately uncertain.

## 🏗️ Architecture

### Custom TrigConv2D Layer

Instead of random initialization, the first convolutional layer uses **fixed trigonometric kernels**:

```python
# Even filters: sin(frequency × (x + y))
# Odd filters:  cos(frequency × (x + y))
```

**Why?** This encodes structured spatial frequency information from the start — similar to positional encoding in transformers, but for images.

```
Input (128×128×3)
       ↓
┌─────────────────┐
│   TrigConv2D    │  ← Sin/Cos frequency kernels (no learned weights)
│   16 filters    │
└────────┬────────┘
         ↓
┌─────────────────┐
│    Conv2D       │  ← Standard learned convolution
│   32 filters    │
└────────┬────────┘
         ↓
    MaxPooling
         ↓
    Dense(64)
         ↓
    Dense(6)  → Softmax
```

---

## 📁 Repository Structure

```
├── notebooks/
│   ├── 02_explainability_demo.ipynb   ⭐ START HERE - Full walkthrough
│   └── public_visualization.ipynb      Additional visualizations
│
├── src/
│   ├── model_trigconv2d.py            TrigConv2D layer definition
│   └── explainability.py              GRAD-CAM & IG implementations
│
├── artifacts/
│   ├── X_test_sample.npy              Test images (4,739 samples)
│   ├── y_test_sample.npy              Test labels
│   ├── label_names.npy                Class name mapping
│   └── trigconv_model.keras           Trained model weights
│
└── assets/                            README images
```

---

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/brain-breast-tumor-ml-classification.git
cd brain-breast-tumor-ml-classification

# Install dependencies
pip install tensorflow numpy matplotlib

# Run the explainability demo
jupyter notebook notebooks/02_explainability_demo.ipynb
```

**No training required** — all artifacts are pre-computed and included.

---

## 📊 Dataset

### Classes & Distribution

| Class | Modality | Description |
|-------|----------|-------------|
| **Benign** | Breast MRI | Non-cancerous breast lesion |
| **Malignant** | Breast MRI | Cancerous breast tumor |
| **No Tumor** | Brain MRI | Healthy brain scan |
| **Glioma Tumor** | Brain MRI | Tumor from glial cells |
| **Meningioma Tumor** | Brain MRI | Tumor from meninges |
| **Pituitary Tumor** | Brain MRI | Tumor in pituitary gland |

### Preprocessing Pipeline
- ✅ BGR → RGB conversion
- ✅ Resize to 128 × 128
- ✅ Normalize to [0, 1]
- ✅ Stratified train/test split (before oversampling)
- ✅ Class balancing via oversampling

---

## 📚 Method Comparison

| Aspect | GRAD-CAM | Integrated Gradients |
|--------|----------|---------------------|
| **Computation** | Fast (single backward pass) | Slower (100 interpolation steps) |
| **Resolution** | Coarse (feature map size) | Fine (pixel-level) |
| **What it shows** | Regional activation | Pixel importance scores |
| **In this model** | Modality discrimination | Tumor-specific features |
| **Clinical use** | Verify anatomical focus | Identify diagnostic regions |

---

## 🔮 Future Work

- [ ] Adversarial robustness testing
- [ ] Confidence calibration curves
- [ ] Additional XAI methods (SHAP, LIME)
- [ ] Deployment as web application

---

## 📖 Citations

**Brain MRI Dataset:**
> Sartaj Bhuvaji, Ankita Kadam, Prajakta Bhumkar, Sameer Dedge, Swati Kanchan. (2020). Brain Tumor Classification (MRI). Kaggle. DOI: 10.34740/KAGGLE/DSV/1183165

**Breast MRI Dataset:**
> Breast MRI dataset from Kaggle medical imaging collection.

---

## 🔒 Code Privacy Note

The complete implementation (training scripts, data pipelines, full `TrigConv2D` implementation) is maintained in a **private repository** for academic integrity.

This public repository provides:
- ✅ Trained model artifacts
- ✅ Explainability demonstrations
- ✅ Reproducible inference notebooks
- ✅ Architecture documentation

**Recruiters & reviewers:** Full codebase available upon request.

---

## 📬 Contact

**Vihari Tejo**

📧 [vihari5tejo@gmail.com](mailto:vihari5tejo@gmail.com)

💼 [LinkedIn](https://linkedin.com/in/yourprofile) • 🐙 [GitHub](https://github.com/yourusername)

---

<div align="center">

**⭐ If this project demonstrates the skills you're looking for, let's connect! ⭐**

</div>
