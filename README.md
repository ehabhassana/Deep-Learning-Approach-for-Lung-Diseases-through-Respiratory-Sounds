# A Hybrid CNN-XGBoost Model with Multi-Feature Fusion and Imbalance Handling for Respiratory Sound Classification and Pulmonary Disease Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📚 Description
This repository contains the official implementation for the paper: 
**"A Hybrid CNN-XGBoost Model with Multi-Feature Fusion and Imbalance Handling for Respiratory Sound Classification and Pulmonary Disease Detection"**.

We provide a complete, end-to-end framework for classifying respiratory sounds (e.g., COPD, Pneumonia, Bronchiolitis) using a hybrid architecture that combines a Convolutional Neural Network (CNN) for feature extraction and XGBoost for robust, non-linear classification.

## ✨ Key Features
*   **Hybrid Architecture**: A 1D-CNN for spectral-temporal feature learning followed by an XGBoost classifier.
*   **Multi-Feature Fusion**: Implementation of feature extraction for MFCC, Energy, Spectral Bandwidth, and their combinations.
*   **Imbalance Handling**: Built-in support for both **SMOTE** oversampling and **class-weighting** strategies.
*   **Robust Validation**: Scripts for **patient‑level stratified 3-fold cross-validation** to prevent data leakage.
*   **Comprehensive Metrics**: Calculates MCDM scores, generates confusion matrices, ROC curves, and Precision-Recall curves.
*   **Reproducibility**: Fixed random seeds and versioned dependencies to ensure results are reproducible.

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/ehabhassana/RespiratorySoundClassifier.git
cd RespiratorySoundClassifier
