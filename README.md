# Brain and Breast Tumor ML Classification Model

A multimodal medical imaging classification model that uses a custom **TrigConv2D layer** to classify brain and breast tumor MRI scans.

⚠️ **Note**: Model training code and preprocessing logic are kept private. This repository contains:
- Public documentation and architecture details
- Dataset summaries and class distributions
- Preprocessed artifacts for demonstration
- Explainability visualizations (Grad-CAM, Integrated Gradients)
- Inference and visualization notebooks

## 🏗️ Architecture

The model uses a custom **TrigConv2D** layer that incorporates trigonometric transformations for enhanced feature extraction from medical images. See [src/trigconv2d.py](src/trigconv2d.py) for implementation details.

## 📁 Repository Structure

```
brain-breast-tumor-ml-classification/
├── data/
│   ├── raw/                      # Raw image paths (not included in public repo)
│   ├── processed/                # Full preprocessed datasets (.npy files)
│   ├── brain_data.md             # Brain tumor dataset documentation
│   ├── breast_data.md            # Breast cancer dataset documentation
│   └── class_summary.md          # Class distribution summaries
│
├── artifacts/                    # Small public inference samples
│   ├── X_test_sample.npy         # Sample test images (50 samples)
│   ├── y_test_sample.npy         # Sample test labels
│   ├── label_names.npy           # Class label names
│   ├── history.json              # Training history (optional)
│   └── trigconv_model.keras      # Trained model (if available)
│
├── src/                          # Source code
│   ├── trigconv2d.py             # Custom TrigConv2D layer
│   ├── model_trigconv2d.py       # Model architecture
│   ├── explainability.py         # Grad-CAM & Integrated Gradients
│   ├── train.py                  # Training script
│   ├── eval.py                   # Evaluation utilities
│   └── preprocessing.py          # Preprocessing stub (private logic not included)
│
├── notebooks/
│   ├── private_preprocessing.ipynb    # Generates artifacts (NOT in public repo)
│   ├── public_visualization.ipynb     # Public demo using artifacts ✅
│   └── model_training.ipynb           # Model training notebook
│
├── docs/                         # Additional documentation
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install tensorflow numpy matplotlib pandas scikit-learn
```

### Running the Public Visualization Notebook

The public visualization notebook demonstrates the model's capabilities using preprocessed artifacts:

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/brain-breast-tumor-ml-classification.git
   cd brain-breast-tumor-ml-classification
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt  # if available
   ```

3. **Open the public notebook**
   ```bash
   jupyter notebook notebooks/public_visualization.ipynb
   ```

4. **Run all cells**
   - The notebook will load preprocessed sample data from `artifacts/`
   - No raw data or private preprocessing is required
   - Visualizations include:
     - Sample medical images
     - Class distribution analysis
     - Grad-CAM explainability overlays
     - Integrated Gradients attribution maps

## 📊 What's Included

### Preprocessed Artifacts

The `artifacts/` folder contains:
- **X_test_sample.npy**: 50 preprocessed test images
- **y_test_sample.npy**: Corresponding one-hot encoded labels
- **label_names.npy**: Array of class names
- **history.json**: Training history metrics (if available)
- **trigconv_model.keras**: Trained model weights (if available)

### Explainability

The model includes two explainability techniques:

1. **Grad-CAM (Gradient-weighted Class Activation Mapping)**
   - Highlights which regions of the image influence predictions
   - Visual heatmaps overlaid on original images

2. **Integrated Gradients**
   - Pixel-level attribution showing importance of each pixel
   - More fine-grained than Grad-CAM

See [notebooks/public_visualization.ipynb](notebooks/public_visualization.ipynb) for examples.

## 🔒 Private Components

The following are kept private for security and proprietary reasons:
- Raw medical imaging data
- Full preprocessing pipeline
- Complete training datasets
- Detailed training scripts with hyperparameters

## 📈 Model Performance

Training and evaluation metrics can be visualized in the public notebook if `history.json` is available in the artifacts folder.

## 🤝 Contributing

This is a demonstration repository. For questions or collaboration inquiries, please open an issue.

## 📄 License

[Add your license here]

## 🔗 Related Links

- [TrigConv2D Layer Documentation](src/trigconv2d.py)
- [Explainability Methods](src/explainability.py)
- [Dataset Summaries](data/class_summary.md) 
