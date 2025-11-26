# Brain Tumor MRI Detection
Classical Computer Vision + Deep Learning (ResNet-18)

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Contributions Welcome](https://img.shields.io/badge/Contributions-welcome-orange)

## Quick Links
Final Report: https://github.com/arianazapata/brain-tumor-mri-detection/blob/f5b9b74a4270754196598f9efc1e85b87ac1cd95 Brain_Tumor_MRI_Detection_Report.pdf
Dataset (Kaggle): https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection  
Repository: https://github.com/arianazapata/brain-tumor-mri-detection

## Motivation
Early and accurate detection of brain tumors significantly improves patient outcomes. However, manually reviewing MRI scans is time-consuming and subjective. This project investigates the performance of two approaches:

- Classical computer vision (thresholding, morphology, feature extraction + logistic regression)
- Deep learning (transfer learning with ResNet-18)

This topic is also personally meaningful. My mom was diagnosed with stage 4 breast cancer at age 40, when I was about four years old. Growing up around that influenced my interest in medical technology and inspired me to choose this project and career path focused on healthcare and early detection tools.

## Project Overview
### Classical Computer Vision Pipeline
1. Grayscale normalization  
2. Gaussian blur  
3. Otsu thresholding  
4. Morphological opening and closing  
5. Largest connected component extraction  
6. Feature extraction (area, bounding box dimensions, aspect ratio, mean intensity)  
7. Logistic Regression classifier  

### CNN Pipeline (ResNet-18 Transfer Learning)
1. Convert MRI grayscale image to 3-channel  
2. Resize to 224x224  
3. ImageNet normalization  
4. Replace the final fully connected layer for two classes  
5. Train for 5 epochs  
6. Evaluate on the test set  

## Results
| Model            | Accuracy | No-Tumor Recall | Tumor Recall |
|------------------|----------|------------------|--------------|
| Classical CV     | ~70%     | ~64%             | ~75%         |
| ResNet-18 CNN    | ~81%     | ~55%             | 100%         |

## Folder Structure
```
brain-tumor-mri-detection/
│
├── data/                # (ignored) train/val/test splits
├── data_raw/            # (ignored) raw dataset
├── preprocessed/        # (ignored) masks, edges, resized images
├── results/             # (ignored) metrics, features
├── models/              # (ignored) saved .pth weights
│
├── src/
│   ├── preprocess.py
│   ├── features.py
│   ├── classical_model.py
│   ├── cnn_model.py
│   ├── evaluate.py
│   └── utils.py
│
├── environment.yml
├── requirements.txt
├── README.md
└── LICENSE
```

## Requirements and Setup
### Environment Creation
```
conda env create -f environment.yml
conda activate braincv2
```

### Pip-only Setup
```
pip install -r requirements.txt
```

## Running the Pipeline
### Preprocess images
```
python src/preprocess.py
```

### Extract features for classical model
```
python src/features.py
```

### Train classical model
```
python src/classical_model.py
```

### Train CNN model
```
python src/cnn_model.py
```

### Evaluate models
```
python src/evaluate.py
```

## Dataset
Dataset: Brain MRI Images for Brain Tumor Detection  
https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection

1. Download the ZIP from Kaggle.  
2. Extract to:
```
data_raw/brain_tumor_dataset/
```
3. Run preprocessing scripts to generate train, val, and test splits.

## Usage Example
```
import torch
from torchvision import transforms
from PIL import Image
from src.utils import load_model

model = load_model("models/cnn_best.pth")
img = Image.open("example.jpg").convert("L")

tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

x = tf(img).unsqueeze(0)
out = model(x)
_, label = torch.max(out, 1)
print(["no_tumor", "tumor"][label])
```

## Future Work
- Data augmentation  
- Fine-tuning deeper layers  
- Grad-CAM interpretability  
- Larger dataset and external validation  
- Reduce false positives using weighted loss  

## License
MIT License
