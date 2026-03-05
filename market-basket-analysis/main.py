import os
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1️⃣ Ensure folder
os.makedirs("market-basket-analysis", exist_ok=True)

# 2️⃣ Load dataset (if not exists, create synthetic realistic)
dataset_path = "market-basket-analysis/transactions.csv"
if not os.path.exists(dataset_path):
    print("Dataset not found. Creating synthetic dataset...")
    np.random.seed(42)
    items = ['Milk','Bread','Butter','Cheese','Eggs','Cereal','Juice','Fruits','Yogurt','Water','Chocolate','Soda','Snacks']
    transactions = []
    for _ in range(1000):  # Increased transactions for better rules
        n_items = np.random.randint(1,6)  # 1–5 items per transaction
        transactions.append(list(np.random.choice(items, n_items, replace=False)))
    df = pd.DataFrame(transactions, columns=[f'Item{i+1}' for i in range(5)])
    df.to_csv(dataset_path, index=False)
else:
    df = pd.read_csv(dataset_path)
    print("Dataset loaded from file.")

print(df.head())

# 3️⃣ One-hot encoding for Apriori
all_items = set(df.values.flatten()) - {np.nan}
ohe_df = pd.DataFrame()
for item in all_items:
    ohe_df[item] = df.apply(lambda row: item in row.values, axis=1).astype(bool)

# 4️⃣ Frequent itemsets using Apriori
freq_itemsets = apriori(ohe_df, min_support=0.05, use_colnames=True)
print("\nFrequent Itemsets:")
print(freq_itemsets.sort_values('support', ascending=False).head(10))

# 5️⃣ Association rules
rules = association_rules(freq_itemsets, metric='lift', min_threshold=1.0)  # lowered threshold
if rules.empty:
    print("\nNo strong association rules found with current thresholds.")
else:
    print("\nTop Association Rules:")
    print(rules.sort_values('lift', ascending=False).head(10))

# 6️⃣ Visualizations
if not rules.empty:
    # Support vs Confidence scatter
    plt.figure(figsize=(8,6))
    sns.scatterplot(x='support', y='confidence', size='lift', data=rules, alpha=0.7, legend=False)
    plt.title("Support vs Confidence (size=Lift)")
    plt.xlabel("Support")
    plt.ylabel("Confidence")
    plt.show()

    # Top 10 lifts barplot
    top_lifts = rules.sort_values('lift', ascending=False).head(10)
    plt.figure(figsize=(10,6))
    sns.barplot(x='lift', y=top_lifts.index, data=top_lifts)
    plt.title("Top 10 Association Rules by Lift")
    plt.show()