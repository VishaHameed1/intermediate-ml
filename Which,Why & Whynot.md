
---

## 🧠 Technical Design Decisions (FAQ)

### 1. 🖼️ CNN Image Classifier

* **Model Used:** **CNN (Convolutional Neural Network)**
* **Why:** CNNs use "filters" to detect spatial patterns (edges, textures). It understands that pixels close to each other form a feature (like an ear or a tail).
* **Why NOT an ANN?** A standard Artificial Neural Network flattens the image, losing the "shape" of the object. Also, a 200x200 image would create 40,000 input nodes, making the model too computationally expensive and prone to overfitting.

---

### 2. 📉 Customer Churn Prediction

* **Model Used:** **Logistic Regression**
* **Why:** Churn is a Binary Classification problem. Logistic Regression provides a "probability score" (0 to 1), making it easy for businesses to set thresholds for "High Risk" customers.
* **Why NOT Linear Regression?** Linear Regression predicts continuous values (e.g., 1.5 or -0.5), which don't make sense for a "Yes/No" outcome. It cannot handle the categorical boundaries of churn.

---

### 3. 👥 Customer Segmentation

* **Model Used:** **K-Means Clustering**
* **Why:** This is "Unsupervised Learning." Since we don't have pre-defined labels, K-Means groups customers based on the mathematical distance (Euclidean distance) between their spending habits.
* **Why NOT KNN?** K-Nearest Neighbors (KNN) is a "Supervised" model; it requires labeled data to know what the groups are. K-Means creates the groups itself from raw data.

---

### 4. 💳 Fraud Detection

* **Model Used:** **Random Forest**
* **Why:** Fraud data is extremely "Imbalanced" (0.1% fraud vs. 99.9% normal). Random Forest uses an ensemble of many trees to ensure rare fraud cases aren't ignored by the majority class.
* **Why NOT a Simple Decision Tree?** A single tree "overfits" easily. It might memorize specific transaction amounts rather than learning general fraudulent behavior patterns.

---

### 5. 💰 Loan Default Prediction

* **Model Used:** **XGBoost**
* **Why:** XGBoost is the industry standard for tabular data. It uses Gradient Boosting, where each new tree fixes the errors made by the previous one, leading to very high accuracy.
* **Why NOT a Neural Network?** Deep Learning usually requires massive datasets to outperform XGBoost. For structured spreadsheet data, XGBoost is faster, more efficient, and easier to tune.

---

### 🛒 6. Market Basket Analysis

* **Model Used:** **Apriori Algorithm**
* **Why:** Specifically designed for "Association Rule Mining." It calculates **Lift** to prove that the relationship between "Bread" and "Butter" is a real pattern, not just a random coincidence.
* **Why NOT Clustering?** Clustering tells you which customers are similar, but it won't give you the specific "If-Then" rules (e.g., "If they buy A, they will buy B") needed for store inventory.

---

### 🎬 7. Movie Recommender Pro

* **Model Used:** **Collaborative Filtering**
* **Why:** It leverages "User Similarity." If User A and User B have similar tastes, the model recommends movies User B liked to User A. It discovers items without needing detailed movie descriptions.
* **Why NOT Content-Based?** Content-based filtering only suggests movies *exactly* like what you've seen (e.g., only Action). Collaborative Filtering introduces users to new genres based on the "wisdom of the crowd."

---

### 📈 8. Stock Price Prediction

* **Model Used:** **LSTM (Long Short-Term Memory)**
* **Why:** Stocks are "Sequential." Yesterday’s price affects today’s. LSTMs have a "Memory Cell" that remembers long-term trends and ignores short-term noise.
* **Why NOT a standard RNN?** Standard RNNs suffer from "Vanishing Gradient," meaning they forget data from a few days ago. LSTMs can remember patterns from weeks or months back.

---

