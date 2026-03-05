
---

## 2️⃣ Customer Churn Prediction – Telecom / SaaS

**`intermediate-ml/customer-churn/README.md`**

```markdown
# Customer Churn Prediction (Telecom / SaaS)

## Overview
Predicts whether a customer will churn using Random Forest.  
Dataset contains customer info, services, and billing data.

## Dataset
- Source: [Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)
- Features: 20+ columns (categorical + numeric)
- Target: `Churn` (Yes/No)

## Key Steps
1. Load dataset with `pandas`.
2. Convert `TotalCharges` to numeric.
3. Encode categorical variables using `LabelEncoder`.
4. Train/test split (`stratify=y`).
5. Standardize features using `StandardScaler`.
6. Train Random Forest classifier.
7. Evaluate using:
   - Confusion Matrix
   - Classification Report
   - ROC-AUC Score
8. Visualizations:
   - Churn distribution
   - Confusion Matrix heatmap
   - Feature importance

## How to Run
```bash
python main.py