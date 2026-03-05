
---

## 3️⃣ Loan Default Prediction

**`intermediate-ml/loan-default/README.md`**

```markdown
# Loan Default Prediction

## Overview
Predicts loan default risk using Random Forest with SMOTE to handle imbalanced data.

## Dataset
- Source: [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data)
- Features: Credit and financial info
- Target: `SeriousDlqin2yrs` (0 = No Default, 1 = Default)

## Key Steps
1. Load dataset with `pandas`.
2. Drop unnecessary ID columns.
3. Train/test split (`stratify=y`).
4. Handle class imbalance using SMOTE.
5. Standardize features.
6. Train Random Forest classifier.
7. Evaluate using:
   - Confusion Matrix
   - Classification Report
   - ROC-AUC Score
8. Visualizations:
   - Default distribution
   - Confusion Matrix heatmap
   - Feature importance

## How to Run
```bash
python main.py