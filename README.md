# Driver Inattention Detection

This repository contains all code, models, and training workflows for detecting **driver inattention** using:
- Traditional ML (SVM, KNN, Random Forest, Naive Bayes, Decision Tree)
- Deep Learning (CNN + Attention, ASTN Spatio-Temporal Model)

---

## 📂 Project Structure

The folder layout of this repo is:

├── .vscode/
├── Codes/
├── src/
│ ├── init.py
│ ├── data_loader.py
│ ├── feature_extractor.py
│ ├── ml_traditional.py
│ ├── models.py
│ └── train.py
├── .gitignore
├── README.md
└── requirements.txt

---

## 🚀 Quick Start (Colab)

Install necessary packages and download the dataset:

!pip install opendatasets
import opendatasets as od

od.download("https://www.kaggle.com/datasets/zeyad1mashhour/driver-inattention-detection-dataset")

---

Training Classical ML Models

from src.ml_traditional import run_traditional_ml_models

---

ASTN — Attention-Based Spatio-Temporal Network
from src.models import create_proposed_astn_model
