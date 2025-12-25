# Multi-Modal Tumor Classification with Explainability  
**Brain + Breast MRI | TrigConv2D Feature Layer | Grad-CAM & Integrated Gradients**

This project explores how a convolutional neural network can classify **six diagnostic categories across two imaging modalities** — brain MRI and breast MRI — while **showing *how* it makes decisions**, not just *what* it predicts.

Because the full model code is private, **this repository provides the public-facing components needed to evaluate the work**:
- architecture diagrams  
- dataset summaries  
- explainability demonstrations  
- evaluation artifacts  
- sample inference notebook

> **Goal:** show the reasoning behind predictions — especially how different attribution methods (Grad-CAM vs. Integrated Gradients) highlight *structure-level vs pixel-level* signals.

---

## Project Scope

| Task | Description |
|------|-------------|
| **Classification** | Predict one of 6 classes — across *two* MRI modalities |
| **Custom Layer** | Uses a **TrigConv2D** feature extractor based on sin/cos kernels |
| **Explainability** | Grad-CAM for broad attention, IG for pixel-level reasoning |
| **Design Focus** | Understanding *data flow* + *model reasoning* without exposing private code |
| **Artifacts** | `.npy` samples, checkpoints, attribution visualizations, training curves |

---

## Why Multi-Modal Matters

The model develops an internal two-step logic:

1️ *Recognize the modality* → brain vs. breast  
2️ *Predict the diagnosis* → one of six target classes  

Even without seeing the private source code, **explainability results reveal this hierarchy** —  
Grad-CAM highlights modality-level structures, while IG isolates finer regions related to tumor identity.


---

## Classes & Modalities

| Class | Modality |
|--------|----------|
| Benign | Breast |
| Malignant | Breast |
| No Tumor | Brain |
| Glioma Tumor | Brain |
| Meningioma Tumor | Brain |
| Pituitary Tumor | Brain |

---

## TrigConv2D Feature Layer

Instead of beginning with random convolutional filters, the model starts with:
- **sinusoidal kernels** (even filters)
- **cosine kernels** (odd filters)

This encourages the network to **encode structured spatial variation earlier**, rather than relying entirely on learned weights.

> *Think positional encoding for images — but built into the first convolution.*

---

## Explainability Case Studies

| Case | Scan | Insight |
|------|------|---------|
| **1 — Breast, Benign** | IG highlights tumor-relevant intensity regions | 
| **2 — Brain, Glioma** | Grad-CAM: broad spatial attention; IG: detailed attribution |
| **3 — Misleading Visual Perception** | Organs may appear to be different than they actually are to the untrained human eye |
| **4 — Low Confidence Case** | Diffuse explanations mirror uncertain predictions |

**Key Idea**

| Method | What it highlights |
|--------|-------------------|
| **Grad-CAM** | *Where* the model is generally looking — broad structural focus |
| **Integrated Gradients** | *Which pixels* shift the prediction — diagnostic reasoning |

> Full walkthrough → `notebooks/02_explainability_demo.ipynb`  
(runs directly on saved artifacts)

---

## Sample Results

<div align="center">

| Method | Visualization |
|--------|--------------|
| Grad-CAM Overlay | ![gradcam](figures/gradcam_example.png) |
| IG Overlay | ![ig](figures/ig_example.png) |

</div>

---

## Dataset Citations

**Brain MRI dataset**
Sartaj Bhuvaji, Ankita Kadam, Prajakta Bhumkar, Sameer Dedge, Swati Kanchan. (2020).
Brain Tumor Classification (MRI). Kaggle.
DOI:10.34740/KAGGLE/DSV/1183165

**Breast MRI dataset**
Sartaj Bhuvaji, Ankita Kadam, Prajakta Bhumkar, Sameer Dedge, Swati Kanchan. (2020).
Brain Tumor Classification (MRI). Kaggle.
DOI:10.34740/KAGGLE/DSV/1183165


---

## About Code Privacy

The full implementation (including `TrigConv2D`, data pipelines, and training scripts) is stored in a **private repository**.

This public repo is designed to showcase:
- explainability behavior  
- model reasoning  
- reproducible sample inferences  
- dataset documentation  
- evaluation artifacts  

> Reviewers can **verify reasoning** without access to private code.

---

## Future Work

-  adversarial & robustness tests  
-  calibration / confidence curves  

---

##  Contact

For access to the private codebase or technical discussion:
vihari5tejo@gmail.com


---

## Summary

> This project does **not** just classify images.  
> It **shows how a model reasons across imaging modalities** —  
> and how different explainability methods reveal different pieces of that reasoning.

