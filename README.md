# Credit Risk Classification Model

Predicting Probability of Default (PD) using machine learning.

## Purpose

This project demonstrates skills relevant for credit risk modeling in banking:
- Binary classification for default prediction (similar to PD models)
- Model validation using AUC, Gini coefficient, and confusion matrices
- Feature importance analysis for model interpretability
- Understanding business trade-offs (False Positives vs False Negatives)

Built as part of internship application for Credit Risk Modeling (PD/LGD/EAD) role.

## Technologies

- Python 3.x
- scikit-learn (Logistic Regression, Random Forest, cross-validation)
- pandas, numpy (data handling)
- matplotlib, seaborn (visualizations)

## Results

| Model | Accuracy | AUC | Gini |
|-------|----------|-----|------|
| Logistic Regression | ~85-88% | ~0.85-0.90 | ~0.70-0.80 |
| Random Forest | ~88-91% | ~0.90-0.94 | ~0.80-0.88 |

Top 5 most important features (Random Forest):
1. debt_to_income
2. credit_utilization  
3. income_ratio
4. late_payments_30d
5. credit_age_months

## How to Run

```bash
# Install dependencies
pip install numpy pandas scikit-learn matplotlib seaborn

# Run the script
python credit_risk_model.py
