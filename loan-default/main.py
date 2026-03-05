import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# 1️⃣ Ensure folder exists
os.makedirs("loan-default", exist_ok=True)

# 2️⃣ Check dataset
dataset_path = "loan-default/cs-training.csv"
if not os.path.exists(dataset_path):
    print("Kaggle dataset not found. Generating synthetic dataset...")
    np.random.seed(42)
    df = pd.DataFrame({
        'RevolvingUtilizationOfUnsecuredLines': np.random.rand(1000),
        'age': np.random.randint(21,75,1000),
        'NumberOfTime30-59DaysPastDueNotWorse': np.random.randint(0,10,1000),
        'DebtRatio': np.random.rand(1000),
        'MonthlyIncome': np.random.randint(2000,15000,1000),
        'NumberOfOpenCreditLinesAndLoans': np.random.randint(1,15,1000),
        'NumberOfTimes90DaysLate': np.random.randint(0,5,1000),
        'NumberRealEstateLoansOrLines': np.random.randint(0,5,1000),
        'NumberOfTime60-89DaysPastDueNotWorse': np.random.randint(0,5,1000),
        'SeriousDlqin2yrs': np.random.choice([0,1],1000,p=[0.93,0.07])
    })
    df.to_csv(dataset_path, index=False)
    print("Synthetic dataset created.")
else:
    df = pd.read_csv(dataset_path)
    print("Dataset loaded from file.")

# 3️⃣ Quick overview
print(df.head())
print(df.info())
print(df['SeriousDlqin2yrs'].value_counts())

# 4️⃣ Features and target
X = df.drop('SeriousDlqin2yrs', axis=1)
y = df['SeriousDlqin2yrs']

# 5️⃣ Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6️⃣ Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"Original training set: {len(y_train)}, Defaults: {sum(y_train==1)}")
print(f"Resampled training set: {len(y_train_res)}, Defaults: {sum(y_train_res==1)}")

# 7️⃣ Feature scaling
scaler = StandardScaler()
X_train_res = scaler.fit_transform(X_train_res)
X_test = scaler.transform(X_test)

# 8️⃣ Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_res, y_train_res)

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

# Class distribution
plt.figure(figsize=(6,4))
sns.countplot(x=y)
plt.title("Loan Default Distribution (0=No, 1=Yes)")
plt.show()

# Confusion matrix heatmap
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

# ROC Curve
RocCurveDisplay.from_predictions(y_test, y_prob)
plt.title("ROC Curve")
plt.show()