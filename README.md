# Credit Risk Classification Model

Predicting Probability of Default (PD) using machine learning.

## Purpose
Demonstration project for Credit Risk Modeling internship application.
Built to show understanding of:
- Binary classification for default prediction
- Model validation (AUC, Gini, confusion matrix)
- Feature importance analysis
- Business trade-offs (FP vs FN)

## Technologies
- Python (pandas, numpy, scikit-learn)
- Logistic Regression & Random Forest
- Matplotlib & Seaborn for visualizations

## Results
- Logistic Regression AUC: ~0.85-0.90
- Random Forest AUC: ~0.90-0.94
- Top features: income_ratio, debt_to_income, credit_utilization

## How to run
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
python credit_risk_model.py