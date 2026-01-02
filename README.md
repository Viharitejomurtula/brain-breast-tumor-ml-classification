# Multi-Modal Tumor Classification with Explainable AI  
**Brain + Breast MRI Classification | Custom TrigConv2D Architecture | Grad-CAM & Integrated Gradients**

---

## Project Summary

**Not just classification — understanding *how* and *why* a model makes its decisions.**

This project builds a **6-class tumor classifier** across **two MRI modalities (brain & breast)** with a strong emphasis on **explainable AI (XAI)**.  
Rather than treating explainability as an afterthought, the core goal is to **analyze what different explanation methods reveal about model reasoning**.

A key finding: **different XAI tools expose different stages of the model's decision process**.

---

## What I Built & Why It Matters

| Component | Why It Matters |
|---------|----------------|
| **Multi-modal CNN classifier** | Handles anatomically different scan types in a single model |
| **Custom TrigConv2D layer** | Injects structured spatial patterns into early feature extraction |
| **Dual explainability pipeline** | Shows complementary strengths of Grad-CAM vs Integrated Gradients |
| **Low-confidence analysis** | Demonstrates how uncertainty correlates with diffuse explanations |

---

## Key Results

### Model Performance (Slice-Level)
- **Test Accuracy:** 95.95%  
- **Classes:** 6  
- **Input Resolution:** 128 × 128 × 3  
- **Test Samples:** 4,739  

> **Note:** Accuracy is reported at the **slice level**, after class balancing, and is intended for **comparative analysis**, not clinical use.

---

## Core Explainability Insight

> **Key Insight:**  
> **Grad-CAM reveals *where the model attends* at a structural level, while Integrated Gradients reveals *which pixels drive the decision*.**

This distinction becomes especially important in **multi-modal models**, where the network must first identify *what kind of image it is* before classifying *what is in the image*.

---

## How the Model Reasons (High-Level)

The classifier implicitly learns a **two-step hierarchy**:
```
Step 1: Modality Recognition
          ↓
Step 2: Tumor Classification
```

Different explainability methods expose different parts of this hierarchy.

---

## Explainability Method Comparison

| Method | What It Reveals | Resolution | Best Used For |
|------|----------------|------------|---------------|
| **Grad-CAM** | Broad structural attention | Coarse | Verifying modality & anatomy focus |
| **Integrated Gradients** | Pixel-level contribution | Fine | Understanding tumor-specific cues |

---

## Explainability Case Studies

### Case 1: Breast MRI — Benign (Medical Interpretation Focus)

**Observation**
- Grad-CAM highlights broad tissue boundaries and overall structure.
- Integrated Gradients concentrates on localized regions influencing the prediction.

**Interpretation**
- Grad-CAM helps confirm the model correctly recognizes **breast anatomy**.
- Integrated Gradients provides clearer insight into **why the model predicts "benign."**

**Takeaway**  
For understanding *tumor-related reasoning*, Integrated Gradients offers more precise explanations than Grad-CAM.

---

### Case 2: Brain MRI — Glioma Tumor (Technical Interpretation Focus)

**Observation**
- Grad-CAM produces wide, diffuse activation across structural regions.
- Integrated Gradients shows sharper, localized pixel-level attribution.

**Interpretation**
- Grad-CAM reflects how deeper layers aggregate information over large receptive fields.
- Integrated Gradients traces how small pixel changes influence the final output.

**Takeaway**  
Grad-CAM is useful for **architecture-level debugging**, while Integrated Gradients better exposes **decision-driving pixels**.

---

### Case 3: Brain MRI — No Tumor (Human Perception Bias)

**Why This Case Matters**  
Some brain MRI slices can visually resemble breast MRI cross-sections to the human eye due to cropping, contrast, or slice position.

**Observation**
- Grad-CAM emphasizes internal brain structure rather than outer edges.
- Integrated Gradients shows weak, diffuse attribution with no dominant hot spots.

**Takeaway**  
Even when human intuition may be uncertain, the explainability maps show the model is reasoning consistently and not falsely detecting pathology.

---

### Case 4: Low-Confidence Prediction (Model Uncertainty)

**Observation**
- Prediction confidence drops significantly.
- Both Grad-CAM and Integrated Gradients become diffuse and unfocused.

**Interpretation**
- Weak attribution aligns with low confidence.
- The model is not "hallucinating" a strong explanation when unsure.

**Takeaway**  
**Uncertainty and explanation strength move together**, which is a desirable safety property.

---

## Architecture Overview

### Custom TrigConv2D Layer (First Convolution)

Instead of random initialization, the first convolutional layer uses **fixed sine and cosine spatial patterns**.  
This biases early feature extraction toward **structured spatial frequency information**, similar in spirit to positional encodings.

**High-Level Flow**
```
Input (128×128×3)
       ↓
TrigConv2D (fixed sin/cos kernels)
       ↓
Standard Conv2D + Pooling
       ↓
Dense Layers
       ↓
Softmax (6 classes)
```

---

## Repository Structure
```
├──data/
│   └── brain_data.md        ⭐ Start here
│   └── breast_data.md
│   └── class_summary.md
│
├── notebooks/
│   └── 01_model_training.ipynb   
│   └── 02_explainability_demo.ipynb
├── src/
│   └── model_trigconv2d.py            Custom TrigConv2D layer
│   └── explainability.py              Grad-CAM & IG implementations
│   └── train.py                       Training Walkthrough
│   └── trigconv2d.py                  Model Class Definition
│
├── models/
│   └── best_model.keras               Saved Trained Model
│   └── model_card.md                  Model Card
│   └── model_summary.txt              Model SUmmary
│   └── training_metadata.json         Model Metadata
├── artifacts/
│   └── X_test_sample.npy              Sample test images
│   └── y_test_sample.npy              Sample labels
│   └── label_names.npy                Class mapping
│   └── trigconv_model.keras           Trained model artifact
```

---

## Quick Start
```bash
git clone https://github.com/yourusername/brain-breast-tumor-ml-classification.git
cd brain-breast-tumor-ml-classification

pip install tensorflow numpy matplotlib
jupyter notebook notebooks/02_explainability_demo.ipynb
```

---

## Dataset Overview

| Class | Modality |
|-------|----------|
| Benign | Breast MRI |
| Malignant | Breast MRI |
| No Tumor | Brain MRI |
| Glioma | Brain MRI |
| Meningioma | Brain MRI |
| Pituitary | Brain MRI |

### Preprocessing
- Resize to 128 × 128
- Normalize to [0, 1]
- Stratified train/test split
- Class balancing via oversampling

---

## Future Work

- Robustness testing under noise & perturbations
- Confidence calibration curves
- Additional XAI methods (SHAP, LIME)
- Lightweight deployment demo

---

## Citations

**Brain MRI Dataset**  
Sartaj Bhuvaji et al. (2020). Brain Tumor Classification (MRI). Kaggle.  
DOI: 10.34740/KAGGLE/DSV/1183165

**Breast MRI Dataset**  
Bloch et al. (2015). BREAST-DIAGNOSIS. The Cancer Imaging Archive.  
DOI: 10.7937/K9/TCIA.2015.SDNRQXXR

---

## Code & Data Transparency

Core training pipelines and preprocessing code are maintained privately to:
- protect original implementation work
- avoid redistributing licensed datasets
- keep this repo focused on inference & explainability

This public repository provides:
-  Trained model artifacts
-  Reproducible explainability notebooks
-  Architecture & reasoning documentation

Full implementation is available upon request.

---

## Contact

**Vihari Tejo**  
vihari5tejo@gmail.com

💼 [LinkedIn](www.linkedin.com/in/vihari-t-9090982b1) • 🐙 [GitHub](https://github.com/Viharitejomurtula?tab=repositories)

---

⭐ **If this project aligns with what you're looking for, let's connect!**
