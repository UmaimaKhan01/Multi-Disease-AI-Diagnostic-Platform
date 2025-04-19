# Multi-Disease-AI-Diagnostic-Platform
Doctor Dock is a unified deep learning platform for diagnosing five diseases from medical images. Each disease module (COVID-19, Skin Cancer, Brain Tumor, Alzheimer’s, Lung Cancer) has a dedicated pipeline for preprocessing, model training, evaluation, and explainability via Grad-CAM. The repository is organized for clarity and production readiness

```markdown
# Doctor Dock: A Multi-Disease AI Diagnostic Platform

Doctor Dock is a deep learning–powered diagnostic system that detects five critical diseases using medical images. It provides separate, optimized pipelines for each condition, ensuring high accuracy and explainability through Grad-CAM visualizations. This repository focuses on model development and training; UI integration can be added later.

## Diseases Covered

- COVID-19 (Chest X-rays)
- Skin Cancer (Dermoscopic images)
- Brain Tumor (MRI slices)
- Alzheimer’s Disease (3D brain MRI volumes)
- Lung Cancer (Chest X-rays)

## Setup Instructions

1. Clone the repository:

```
git clone https://github.com/umaimakhan01/DoctorDock.git
cd DoctorDock
```

2. Install dependencies:

```
pip install -r requirements.txt
```

Use Python 3.9 or higher. A CUDA-enabled GPU (e.g., RTX 4090) is recommended for training.

## Dataset Placement

Download and extract the following datasets into the `data/` folder:

- `data/covid/` – Kaggle Radiography Database
- `data/skin/` – HAM10000 Dataset
- `data/brain/` – BraTS 2021 Slices
- `data/alzheimer/` – OASIS Dataset
- `data/lung/` – IQ-OTH or NCCD Dataset

- `data/covid/` – [Kaggle Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
- `data/skin/` – [HAM10000 Skin Lesion Dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- `data/brain/` – [BraTS 2021 Brain Tumor Segmentation](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation)
- `data/alzheimer/` – [OASIS-3 Alzheimer's MRI Dataset](https://www.oasis-brains.org/)
- `data/lung/` – [IQ-OTH/NCCD Lung Cancer Dataset](https://www.kaggle.com/datasets/serjlim/iqothnccd-lung-cancer-dataset)

Ensure each disease folder contains appropriate subfolders (e.g., `normal/`, `tumor/`, etc.).

## How to Train

Navigate to the `scripts/` directory and run the relevant script:

```
python scripts/train_covid.py
python scripts/train_skin.py
python scripts/train_brain.py
python scripts/train_alzheimer.py
python scripts/train_lung.py
```

Each script trains the corresponding model, evaluates it, and saves the weights.

## Evaluation

- Accuracy, F1 score, AUC, Dice (for MRI), and IoU are reported.
- Results are printed at the end of each script.
- Classification reports and confusion matrices are included.

## Grad-CAM Visualizations

To visualize model decisions:

1. Open the relevant notebook in `notebooks/`
2. Load a test image
3. Generate the Grad-CAM heatmap overlay

This helps interpret model behavior and improves trust in predictions.

## Model Performance

| Disease      | Accuracy | F1 Score | AUC  |
|--------------|----------|----------|------|
| COVID-19     | 87.2%    | 0.76     | 0.83 |
| Skin Cancer  | 78.6%    | 0.69     | 0.75 |
| Brain Tumor  | 81.4%    | 0.72     | 0.78 |
| Alzheimer’s  | —        | —        | —    |
| Lung Cancer  | 74.3%    | 0.66     | 0.73 |

*Note: Alzheimer’s model training in progress.*

## Frameworks Used

- TensorFlow / Keras
- Scikit-learn
- OpenCV
- Nibabel, SimpleITK
- Matplotlib, Seaborn

## Future Work

- Add UI for end-to-end integration
- Improve Alzheimer’s model accuracy
- Expand dataset sizes
- Enable multi-label and comorbidity prediction

## Contact

**Author**: Umaima Khan, Karthika Ramasamy
**Email**: fn653419@ucf.edu, ka234388@ucf.edu
**Course**: CAP 5516 – Medical Image Computing  
**Instructor**: Dr. Chen Chen

## References

- COVID-Net (Wang et al., 2020)
- ISIC 2018 Skin Lesion Challenge
- BraTS 2021 Tumor Segmentation
- OASIS Alzheimer’s Dataset
- Hussain et al., Lung Cancer Detection
- Grad-CAM: Selvaraju et al., 2017
```
