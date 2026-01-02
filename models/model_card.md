# Model Card — TrigConv2D Multi-Modal MRI Classifier

## Model Overview
- Architecture: CNN with custom TrigConv2D feature layer
- Input: 128×128 RGB MRI slices
- Output: 6-class softmax

## Training Data
- Brain MRI (4 classes)
- Breast MRI (2 classes)

## Intended Use
- Educational and research demonstration
- Explainability analysis
- NOT for clinical diagnosis

## Limitations
- Dataset imbalance
- Scanner variability
- Slice-level inference only

## Explainability
- Grad-CAM: structural focus
- Integrated Gradients: pixel-level attribution

## Ethical Notes
- Predictions should not be used for medical decisions
- Model confidence and explanation quality must be interpreted together
