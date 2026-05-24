"""
Credit Risk Classification Model
Author: Denys Yakovliev
Purpose: Internship application - Credit Risk Modeling
Date: 2026

A binary classification model predicting default risk using synthetic data.
Demonstrates: logistic regression, Random Forest, model validation (AUC, Gini, confusion matrix)
"""

# ============ 1. IMPORTS ============

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# For reproducibility
np.random.seed(42)

print("=" * 60)
print("CREDIT RISK CLASSIFICATION MODEL")
print("Predicting Probability of Default (PD)")
print("=" * 60)

# ============ 2. GENERATE SYNTHETIC DATA ============

print("\n[1] Generating synthetic credit application data...")

# Generate 10,000 samples with 20 features
# - 3 informative features (actually predict default)
# - 2 redundant features (correlated with informative)
# - 15 noise features (random)
X, y = make_classification(
    n_samples=10000,
    n_features=20,
    n_informative=3,
    n_redundant=2,
    n_repeated=0,
    n_classes=2,
    n_clusters_per_class=1,
    flip_y=0.03,  # 3% label noise (realistic for credit data)
    random_state=42
)

# Feature names (for interpretability)
feature_names = [
    "income_ratio", "debt_to_income", "credit_utilization",
    "late_payments_30d", "late_payments_60d", "credit_age_months",
    "num_credit_lines", "inquiry_count_6m", "bankruptcy_flag",
    "employment_years", "residential_stability", "loan_amount_ratio",
    "interest_rate", "collateral_value", "previous_default",
    "feature_16", "feature_17", "feature_18", "feature_19", "feature_20"
][:20]

# Create DataFrame
df = pd.DataFrame(X, columns=feature_names)
df["default"] = y

print(f"   Dataset shape: {df.shape}")
print(f"   Default rate: {df['default'].mean():.2%}")
print(f"   Features: {df.shape[1] - 1} predictors, 1 target")

# ============ 3. EXPLORATORY DATA ANALYSIS ============

print("\n[2] Basic data exploration...")

# Check class balance
print(f"   Class 0 (Non-default): {(df['default'] == 0).sum():,} samples")
print(f"   Class 1 (Default): {(df['default'] == 1).sum():,} samples")

# Summary statistics for key features
key_features = ["income_ratio", "debt_to_income", "credit_utilization", "late_payments_30d"]
print("\n   Summary stats for key features (Default = 1 vs Non-default = 0):")
for feat in key_features:
    default_mean = df[df["default"] == 1][feat].mean()
    nondefault_mean = df[df["default"] == 0][feat].mean()
    print(f"     {feat}: Default={default_mean:.3f} | Non-default={nondefault_mean:.3f}")

# ============ 4. TRAIN/TEST SPLIT ============

print("\n[3] Splitting data into train (70%) and test (30%)...")

X = df.drop("default", axis=1)
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"   Train set: {X_train.shape[0]} samples")
print(f"   Test set: {X_test.shape[0]} samples")
print(f"   Train default rate: {y_train.mean():.2%}")
print(f"   Test default rate: {y_test.mean():.2%}")

# ============ 5. MODEL 1: LOGISTIC REGRESSION ============

print("\n[4] Training Logistic Regression model...")

lr_model = LogisticRegression(
    C=1.0,           # Inverse regularization strength
    class_weight="balanced",  # Handle any class imbalance
    solver="liblinear",
    random_state=42,
    max_iter=1000
)

lr_model.fit(X_train, y_train)

# Predictions
y_pred_lr = lr_model.predict(X_test)
y_pred_proba_lr = lr_model.predict_proba(X_test)[:, 1]

# Metrics
lr_accuracy = accuracy_score(y_test, y_pred_lr)
lr_auc = roc_auc_score(y_test, y_pred_proba_lr)
lr_gini = 2 * lr_auc - 1  # Gini = 2*AUC - 1

print(f"   Logistic Regression Results:")
print(f"     Accuracy: {lr_accuracy:.4f} ({lr_accuracy*100:.2f}%)")
print(f"     AUC: {lr_auc:.4f}")
print(f"     Gini: {lr_gini:.4f}")

# Confusion Matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)
print(f"     Confusion Matrix:")
print(f"       TN: {cm_lr[0,0]:,} | FP: {cm_lr[0,1]:,}")
print(f"       FN: {cm_lr[1,0]:,} | TP: {cm_lr[1,1]:,}")

# ============ 6. MODEL 2: RANDOM FOREST ============

print("\n[5] Training Random Forest model...")

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=50,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

# Metrics
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_auc = roc_auc_score(y_test, y_pred_proba_rf)
rf_gini = 2 * rf_auc - 1

print(f"   Random Forest Results:")
print(f"     Accuracy: {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")
print(f"     AUC: {rf_auc:.4f}")
print(f"     Gini: {rf_gini:.4f}")

# ============ 7. MODEL COMPARISON ============

print("\n[6] Model Comparison:")
print("-" * 50)
print(f"{'Metric':<15} {'Logistic Regression':<20} {'Random Forest':<20}")
print("-" * 50)
print(f"{'Accuracy':<15} {lr_accuracy:.4f} ({lr_accuracy*100:.1f}%){'':<6} {rf_accuracy:.4f} ({rf_accuracy*100:.1f}%)")
print(f"{'AUC':<15} {lr_auc:.4f}{'':<16} {rf_auc:.4f}")
print(f"{'Gini':<15} {lr_gini:.4f}{'':<16} {rf_gini:.4f}")
print("-" * 50)

# Feature Importance (Random Forest)
feature_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": rf_model.feature_importances_
}).sort_values("importance", ascending=False)

print("\n   Top 5 most important features (Random Forest):")
for i, row in feature_importance.head(5).iterrows():
    print(f"     {row['feature']}: {row['importance']:.4f}")

# ============ 8. CROSS-VALIDATION ============

print("\n[7] Cross-validation (5-fold) on Logistic Regression...")

cv_scores = cross_val_score(lr_model, X_train, y_train, cv=5, scoring="roc_auc")
print(f"   Cross-validation AUC scores: {cv_scores}")
print(f"   Mean CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# ============ 9. VISUALIZATIONS ============

print("\n[8] Generating visualizations...")

# ROC Curves
plt.figure(figsize=(10, 6))

fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_proba_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)

plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {lr_auc:.3f})", linewidth=2)
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {rf_auc:.3f})", linewidth=2)
plt.plot([0, 1], [0, 1], "k--", label="Random Classifier (AUC = 0.5)", linewidth=1)

plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12)
plt.title("ROC Curves - Credit Risk Default Prediction", fontsize=14)
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150)
print("   Saved: roc_curves.png")

# Confusion Matrix Heatmap (Logistic Regression)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non-Default", "Default"],
            yticklabels=["Non-Default", "Default"])
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Actual", fontsize=12)
plt.title("Confusion Matrix - Logistic Regression", fontsize=14)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("   Saved: confusion_matrix.png")

# Feature Importance Plot
plt.figure(figsize=(10, 6))
plt.barh(feature_importance.head(10)["feature"], feature_importance.head(10)["importance"])
plt.xlabel("Importance", fontsize=12)
plt.title("Top 10 Feature Importances - Random Forest", fontsize=14)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
print("   Saved: feature_importance.png")

# ============ 10. BUSINESS INTERPRETATION ============

print("\n" + "=" * 60)
print("BUSINESS INTERPRETATION")
print("=" * 60)

print("""
For credit risk modeling in banking:

1. AUC (Area Under ROC Curve):
   - 0.5 = random guessing (useless model)
   - 0.7-0.8 = acceptable for credit scoring
   - 0.8-0.9 = good discrimination
   - >0.9 = excellent (rare in real credit data)

2. Gini coefficient (2*AUC - 1):
   - Banks often use Gini to rank models
   - Higher Gini = better separation between defaulters and non-defaulters

3. Confusion Matrix tells us:
   - False Positives (FP): Good clients wrongly flagged as risky
     --> Lost business opportunity (cost of missed revenue)
   - False Negatives (FN): Risky clients missed
     --> Potential loan losses (cost of defaults)

4. In production, banks choose a probability threshold
   balancing:
   - Approval rate (want to say YES to good clients)
   - Default rate (want to say NO to bad clients)
""")

print("\n" + "=" * 60)
print("MODEL READY FOR DEPLOYMENT CONSIDERATION")
print("=" * 60)
print("""
Next steps for real-world implementation:
- Use real historical loan data (not synthetic)
- Add more features (behavioral, macroeconomic)
- Calibrate probabilities to actual default rates
- Validate on out-of-time samples
- Document model development (Model Development Document - MDD)
- Present to Model Risk Management (MRM) for approval
""")

print("\n✅ Credit risk model completed successfully!")
