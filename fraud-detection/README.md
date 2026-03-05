# Fraud Detection (Credit Card)

## Overview
This project builds an end-to-end **credit card fraud detection model** using a Random Forest classifier.  
It uses the `creditcard.csv` dataset and predicts whether a transaction is fraudulent or not.

## Dataset
- Source: Kaggle [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Features: 30 numerical features (V1–V28, Amount, Time)
- Target: `Class` (0 = non-fraud, 1 = fraud)

## Key Steps
1. Load dataset using `pandas`.
2. Split features (`X`) and target (`y`).
3. Train/test split with stratification: `stratify=y`.
4. Train Random Forest classifier.
5. Evaluate using:
   - Confusion Matrix
   - Classification Report

## How to Run
```bash
python main.py