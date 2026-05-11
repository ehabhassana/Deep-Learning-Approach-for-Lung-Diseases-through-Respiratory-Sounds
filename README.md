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

## Dataset

This project uses the **ICBHI 2017 Respiratory Sound Database** (often referred to in some repos as `ICBHI-master`).
The dataset is **not included** in this repository. Please download it from the official source (when available) or a public mirror and place it in the expected directory.

**Official challenge page (may be unavailable at times):**
- http://bhichallenge.med.auth.gr/ICBHI_2017_Challenge
- (Archive) https://web.archive.org/web/*/bhichallenge.med.auth.gr/ICBHI_2017_Challenge

**Public mirror (download):**
- https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database

### How to set up
1. Download the dataset.
2. Extract it to: `data/ICBHI-master/`  (or rename the extracted folder to `ICBHI-master` to match the code).
3. Verify the audio files and annotation files are present.

### Citation
If you use this dataset, please cite the original ICBHI 2017 Respiratory Sound Database publication and the ICBHI 2017 challenge resources (see the dataset page for the recommended citation).

| Experiment No. | Model |
|---:|---|
| 1 | MFCCCNN |
| 2 | MFCCCNN XGBOST |
| 3 | ZCRCNN XGBOST |
| 4 | ZCRCNN |
| 5 | EnergyCNN |
| 6 | Spectral-CentroidCNN |
| 7 | Spectral-BandwidthCNN |
| 8 | Spectral-ContrastCNN |
| 9 | LPCCCNN |
| 10 | PLPCNN |
| 11 | DWTCNN |
| 12 | Mel-SpectrogramCNN |
| 13 | Chroma-FeaturesCNN |
| 14 | CQTCNN |
| 15 | EnergyCNN XGBOST |
| 16 | Spectral-CentroidCNN XGBOST |
| 17 | Spectral-BandwidthCNN XGBOST |
| 18 | Spectral-ContrastCNN XGBOST |
| 19 | LPCCCNN XGBOST |
| 20 | PLPCNN XGBOST |
| 21 | DWTCNN XGBOST |
| 22 | Mel-SpectrogramCNN XGBOST |
| 23 | Chroma-FeaturesCNN XGBOST |
| 24 | CQTCNN XGBOST |
| 25 | WAVCNN XGBOST |
| 26 | SpectromCNN XGBOST |
| 27 | MFCCCNNRNNs |
| 28 | MFCCCNNTransformer |
| 29 | MFCCCNN SVM |
| 30 | MFCCCNN LightGBM |
| 31 | MFCCCNN Random Forest |
| 32 | MFCCCNN Autoencoder |
| 33 | MFCCCNN Attention Mechanism |
| 34 | MFCCCNN Ensemble |
| 35 | MFCC CNN XGBOST Class Imbalance |
| 36 | MFCC Spectral FeaturesCNN XGBOST |
| 37 | MFCC Spectral FeaturesCNN XGBOST class Imbalance |
| 38 | ZCR Spectral CentroidCNN XGBOST |
| 39 | Energy Spectral BandwidthCNN XGBOST |
| 40 | Energy Spectral CentroidCNN XGBOST |
| 41 | ZCR Spectral BandwidthCNN XGBOST |
| 42 | ZCR ENERGYCNN XGBOST |
| 43 | ENERGY MFCCCNN XGBOST |
| 44 | ZCR MFCCCNN XGBOST |
| 45 | Spectral Centroid MFCCCNN XGBOST |
| 46 | Spectral Bandwidth MFCCCNN XGBOST |
| 47 | MFCC DWTCNN XGBOST |
| 48 | Mel-Spectrogram DWTCNN XGBOST |
| 49 | Spectrogram Chroma FeatureCNN XGBOST |
| 50 | ZCR Spectral Centroid MFCCCNN XGBOST |
| 51 | ZCR Spectral Bandwidth MFCCCNN XGBOST |
| 52 | Energy Spectral Bandwidth MFCCCNN XGBOST |
| 53 | Energy Spectral Bandwidth MFCCCNN XGBOST Class Imbalance |
| 54 | Energy Spectral Bandwidth MFCCCNN XGBOST SMOTE |
| 55 | ENERGY MFCC DWTCNN XGBOST |
| 56 | ZCR MFCC DWTCNN XGBOST |
| 57 | Spectral Centroid MFCC Mel-SpectrogramCNN XGBOST |
| 58 | Spectral Bandwidth MFCC Mel-SpectrogramCNN XGBOST |
| 59 | ZCR Spectral Centroid MFCC DWTCNN XGBOST |
| 60 | Energy Spectral Bandwidth MFCC DWTCNN XGBOST |
| 61 | ZCE Spectral Bandwidth MFCC DWTCNN XGBOST |
| 62 | Energy Spectral Centroid MFCC DWTCNN XGBOST |
| 63 | ZCR Spectral Centroid MFCC Mel-SpectrogramCNN XGBOST |
| 64 | ZCR Spectral Bandwidth MFCC Mel-SpectrogramCNN XGBOST |
| 65 | ZCR Spectral Bandwidth MFCC Mel-Spectrogram UMAPCNN XGBOST |
| 66 | Energy Spectral Bandwidth MFCC Mel-SpectrogramCNN XGBOST |
| 67 | Energy Spectral Centroid MFCC Mel-SpectrogramCNN XGBOST |
| 68 | ZCR Spectral Centroid MFCC DWT Chroma FeaturesCNN XGBOST |
| 69 | ZCR Spectral Bandwidth MFCC DWT Chroma FeaturesCNN XGBOST |
| 70 | Energy Spectral Centroid MFCC DWT Chroma FeaturesCNN XGBOST |
| 71 | Energy Spectral Bandwidth MFCC DWT Chroma FeaturesCNN XGBOST |
| 72 | Energy Spectral Bandwidth MFCC DWT Chroma FeaturesCNN XGBOST SMOTE |
| 73 | Energy Spectral Bandwidth MFCC PCACNN XGBOST |
| 74 | Energy Spectral Bandwidth MFCC LDACNN XGBOST |
| 75 | Energy Spectral Bandwidth MFCC t-SNE UMAPCNN XGBOST |

