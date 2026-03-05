# Customer Segmentation (Clustering)

## Overview
This project performs **customer segmentation** using unsupervised learning.  
It uses **K-Means clustering** to group customers based on behavior, spending patterns, and demographics.  

Segmentation can help in:
- Targeted marketing
- Personalized promotions
- Customer retention strategies

## Dataset
- Source: Synthetic or public dataset like `Mall_Customers.csv`
- Columns:
  - `CustomerID` – Unique ID
  - `Gender` – Male/Female
  - `Age` – Customer age
  - `Annual Income (k$)` – Income in thousands
  - `Spending Score (1-100)` – Customer spending score

## Key Steps
1. Load dataset using `pandas`.
2. Explore data:
   - Missing values
   - Summary statistics
3. Encode categorical features (Gender → numeric)
4. Feature scaling using `StandardScaler`
5. Dimensionality reduction (optional) using PCA
6. Apply K-Means clustering:
   - Find optimal number of clusters using **Elbow Method** or **Silhouette Score**
7. Visualize clusters in 2D or 3D

## How to Run
```bash
python main.py