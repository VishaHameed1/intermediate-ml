import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# 1️⃣ Ensure folder
os.makedirs("customer-segmentation", exist_ok=True)

# 2️⃣ Load dataset (if not exists, create synthetic)
dataset_path = "customer-segmentation/customers.csv"
if not os.path.exists(dataset_path):
    print("Dataset not found. Creating synthetic dataset...")
    np.random.seed(42)
    df = pd.DataFrame({
        'Age': np.random.randint(18,70,500),
        'AnnualIncome': np.random.randint(20000,150000,500),
        'SpendingScore': np.random.randint(1,100,500),
        'Savings': np.random.randint(1000,50000,500)
    })
    df.to_csv(dataset_path, index=False)
else:
    df = pd.read_csv(dataset_path)
    print("Dataset loaded from file.")

print(df.head())

# 3️⃣ Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# 4️⃣ PCA (optional 2D for visualization)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 5️⃣ KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels
df['Cluster'] = clusters

# 6️⃣ Visualizations

# PCA scatter plot
plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=clusters, palette="Set2", s=100)
plt.title("Customer Segmentation (PCA 2D)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()

# Cluster-wise boxplots
for col in df.columns[:-1]:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='Cluster', y=col, data=df)
    plt.title(f"{col} by Cluster")
    plt.show()

# 7️⃣ Cluster sizes
print("Cluster counts:")
print(df['Cluster'].value_counts())