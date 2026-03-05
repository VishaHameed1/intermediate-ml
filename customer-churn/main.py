import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1️⃣ Load dataset
df = pd.read_csv("customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# 2️⃣ Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# 3️⃣ Drop customerID
df.drop('customerID', axis=1, inplace=True)

# 4️⃣ Convert categorical columns using get_dummies
cat_cols = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)  # drop_first avoids dummy trap

# 5️⃣ Features and target
X = df.drop('Churn_Yes', axis=1) if 'Churn_Yes' in df.columns else df.drop('Churn', axis=1)
y = df['Churn_Yes'] if 'Churn_Yes' in df.columns else df['Churn']

# 6️⃣ Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7️⃣ Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 8️⃣ Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 9️⃣ Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

# 🔟 Evaluation
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

roc_score = roc_auc_score(y_test, y_prob)
print(f"\nROC-AUC Score: {roc_score:.4f}")

# 1️⃣1️⃣ Visualizations
plt.figure(figsize=(6,4))
sns.countplot(x=y)
plt.title("Churn Distribution (0=No, 1=Yes)")
plt.show()

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Feature importance
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances = feat_importances.sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=feat_importances.values, y=feat_importances.index)
plt.title("Feature Importance")
plt.show()