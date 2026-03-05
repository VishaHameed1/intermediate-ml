import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# 1️⃣ Load dataset
df = pd.read_csv("fraud-detection/creditcard.csv")  # correct relative path

# 2️⃣ Original class distribution
plt.figure(figsize=(6,4))
sns.countplot(x="Class", data=df)
plt.title("Original Class Distribution (0=Non-Fraud, 1=Fraud)")
plt.show()

# 3️⃣ Features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# 4️⃣ Train/Test Split with stratify
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5️⃣ Handle imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"Original training set size: {len(y_train)}, Fraud cases: {sum(y_train==1)}")
print(f"Resampled training set size: {len(y_train_res)}, Fraud cases: {sum(y_train_res==1)}")

# 6️⃣ Visualize resampled data
plt.figure(figsize=(6,4))
sns.countplot(x=y_train_res)
plt.title("Resampled Training Set Class Distribution")
plt.show()

# 7️⃣ Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_res, y_train_res)

# 8️⃣ Predictions
y_pred = model.predict(X_test)

# 9️⃣ Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# 🔟 Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

# 1️⃣1️⃣ Feature Importance
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances = feat_importances.sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=feat_importances.values, y=feat_importances.index)
plt.title("Feature Importance from Random Forest Model")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()