# Aviation Flight Delay Risk Prediction

This repository contains an end-to-end Machine Learning pipeline to predict flight delay risks using the OpenML `airlines` dataset. It covers data cleaning, feature preparation, train/validation/test splitting, baseline and production drift monitoring, model training, threshold optimization, and probability calibration.

## 🚀 Project Overview
* **Goal**: Predict flight delays based on operational features (Airline, AirportFrom, AirportTo, Flight Number, Day of Week, Departure Time, and Flight Length).
* **Models Trained**: 
  * **Baseline**: Dummy Classifier
  * **Linear Baseline**: Logistic Regression (with custom Preprocessing Pipeline)
  * **Advanced Model**: CatBoost Classifier
* **Production Monitoring**: Includes Drift Alert Engine using statistical techniques (Kolmogorov-Smirnov test for continuous features, Chi-Square test with effect size for categorical features, and Two-Proportion Z-test for operational rate shifts).

## 📊 Model Performance Summary

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** (Test) | 0.6978 | 0.6553 | 0.5963 | 0.6167 | 0.6063 |
| **CatBoost (Threshold 0.50)** | 0.7320 | 0.7006 | 0.6791 | 0.5170 | 0.5870 |
| **CatBoost (Optimized Threshold 0.34)** | **0.7320** | **0.7006** | **0.5501** | **0.8201** | **0.6585** |

*Optimizing the CatBoost decision threshold to 0.34 increased the F1-Score on the test set from 0.5870 to 0.6585, significantly boosting recall for flight delay detection.*

## 🛠️ Repository Structure
* `Airlines_Delay_Prediction.ipynb`: Complete Jupyter Notebook containing data preprocessing, evaluation, drift simulation, and modeling steps.
* `requirements.txt`: Python package requirements for reproducibility.
* `README.md`: Project description and performance highlights.

## ⚙️ Installation & Usage
To replicate or run this notebook locally:

1. Clone the repository:
   ```bash
   git clone <your-github-repo-url>
   cd <your-repo-folder>
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook using Jupyter Notebook, JupyterLab, or upload it to Google Colab.