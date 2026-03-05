
---

## 4️⃣ Market Basket Analysis

**`intermediate-ml/market-basket-analysis/README.md`**

```markdown
# Market Basket Analysis

## Overview
Performs association rule mining to find product combinations in transactions using Apriori algorithm.

## Dataset
- Synthetic dataset generated if no CSV file exists.
- Transactions: 1000 rows, 5 items per row on average.
- Items include: Milk, Bread, Butter, Cheese, Eggs, Cereal, Juice, Fruits, Yogurt, etc.

## Key Steps
1. Load dataset (or create synthetic dataset).
2. One-hot encode items for Apriori.
3. Generate frequent itemsets using Apriori (`min_support=0.05`).
4. Generate association rules (`metric='lift', min_threshold=1.0`).
5. Visualizations:
   - Support vs Confidence scatter plot
   - Top 10 rules by lift barplot

## How to Run
```bash
python main.py