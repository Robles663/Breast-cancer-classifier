# Breast Cancer Classifier

Predicting malignancy in breast cancer biopsies using machine learning.

## Overview
This project builds a classification model to predict whether a breast 
tumor is malignant or benign, using the Wisconsin Breast Cancer Dataset 
from scikit-learn. The goal is to demonstrate a full ML workflow with 
clinical context, from EDA to a deployable Streamlit app.

## Objectives
- Perform thorough exploratory data analysis on 30 cell nucleus features
- Build and compare multiple classification models
- Prioritize recall to minimize false negatives (missed malignancies)
- Provide model interpretability via SHAP values
- Deploy an interactive app for risk prediction

## Dataset
- **Source:** Wisconsin Breast Cancer Dataset (sklearn.datasets)
- **Samples:** 569 biopsies
- **Features:** 30 numeric features (mean, SE, worst value of cell characteristics)
- **Target:** Malignant (0) / Benign (1)

## Tech Stack
- Python 3.10+
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- shap
- streamlit

## Project Structure
    breast-cancer-classifier/
    │
    ├── data/
    │   └── raw/
    │       └── breast_cancer.csv
    │
    ├── notebooks/
    │   ├── 01_eda.ipynb
    │   ├── 02_preprocessing.ipynb
    │   ├── 03_modeling.ipynb
    │   ├── 04_evaluation.ipynb
    │   └── models/
    │       ├── final_model.pkl
    │       ├── pca.pkl
    │       └── scaler.pkl
    │
    ├── app/
    │   └── streamlit_app.py
    │
    ├── reports/
    │   └── figures/
    │
    ├── requirements.txt
    └── README.md

## Results

| Model | CV AUC | Test AUC |
|-------|--------|----------|
| Logistic Regression | 0.9951 | 0.9964 |
| SVM | 0.9948 | 0.9964 |
| Random Forest | 0.9875 | 0.9939 |
| KNN | 0.9892 | 0.9897 |

**Final model:** Logistic Regression (C=0.1, solver=liblinear)  
**Test Accuracy:** 98%  
**Recall (Malignant):** 98% — 41 of 42 malignant tumors correctly identified  
**False Negatives:** 1 — one malignant tumor classified as benign

## Run the App
    pip install -r requirements.txt
    streamlit run app/streamlit_app.py

## Limitations
- Dataset is relatively small (569 samples)
- Model should not be used for real clinical decisions
- Results may not generalize to populations outside the original study

## Author
Alejandro Robles  
roblesliz.alejandro@gmail.com

