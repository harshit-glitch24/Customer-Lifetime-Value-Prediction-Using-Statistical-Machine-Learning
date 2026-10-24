# Customer Lifetime Value (CLV) Prediction Using Statistical Machine Learning

<p align="center">
  <strong>A reproducible machine-learning workflow for estimating 12-month customer value</strong>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="scikit--learn" src="https://img.shields.io/badge/scikit--learn-Modeling-F7931E?logo=scikitlearn&logoColor=white">
  <img alt="Status" src="https://img.shields.io/badge/Status-Academic%20Project-informational">
  <img alt="Data" src="https://img.shields.io/badge/Data-Synthetic-orange">
</p>

## Overview

Customer Lifetime Value (CLV) estimates the monetary value a customer is expected to generate over a defined future period. Reliable CLV estimates can help businesses prioritize retention efforts, understand customer segments, and make more informed marketing-allocation decisions.

This project builds an end-to-end statistical machine-learning workflow to predict **customer spend over the next 12 months**. It combines exploratory data analysis, statistical hypothesis testing, regression-model comparison, cross-validation, model diagnostics, feature-importance analysis, and customer-value tier classification.

> **Important data limitation:** The project currently uses a generated synthetic dataset of 2,500 customers. Reported metrics and statistical findings are illustrative demonstrations of the workflow—not evidence of performance on real customer data. Validate on an appropriately governed, representative real-world dataset before using predictions for business decisions.

## Objectives

- Build a repeatable customer-level data preparation workflow.
- Explore customer spending, engagement, recency, tenure, and loyalty characteristics.
- Test whether average future customer value differs between loyalty members and non-members.
- Compare baseline and machine-learning regression models using held-out test data.
- Use cross-validation to assess model stability.
- Inspect influential predictors and regression diagnostics.
- Classify customers into four value tiers for exploratory segmentation.

## Workflow

1. **Data generation / ingestion** — create reproducible synthetic customer records; the notebook can be adapted to load an approved CSV dataset.
2. **Data quality checks** — inspect data types, duplicates, missingness, and target validity.
3. **Exploratory analysis** — review distributions, correlations, customer segments, and recency groups.
4. **Statistical inference** — use Welch's independent two-sample t-test to compare future value for loyalty and non-loyalty groups.
5. **Preprocessing** — median-impute numeric features, most-frequent-impute categorical features, scale numeric predictors, and one-hot encode categorical predictors.
6. **Regression** — compare a mean baseline, multiple linear regression, Ridge regression, and Random Forest regression.
7. **Validation and interpretation** — evaluate MAE, RMSE, R², cross-validation RMSE, residual diagnostics, and permutation importance.
8. **Value-tier classification** — predict Low, Medium-Low, Medium-High, and High CLV tiers and inspect classification metrics/confusion matrix.

## Models and evaluation

| Model | Purpose |
|---|---|
| Dummy Mean Regressor | Baseline for comparison |
| Multiple Linear Regression | Interpretable linear benchmark |
| Ridge Regression | Regularized linear model |
| Random Forest Regressor | Non-linear tree-ensemble benchmark |
| Random Forest Classifier | Exploratory customer-value tier classification |

Regression performance is assessed with:

- **MAE (Mean Absolute Error):** average absolute prediction error, in target currency units.
- **RMSE (Root Mean Squared Error):** penalizes larger errors more strongly.
- **R²:** proportion of target variance explained relative to a mean baseline.
- **Cross-validation RMSE:** checks performance across multiple training folds.

The notebook uses a fixed random seed for reproducibility and separates training and test data. Preprocessing is placed inside scikit-learn pipelines so imputation, scaling, and encoding are learned from training folds rather than the full dataset.

## Repository contents

```text
.
├── Customer_Lifetime_Value_Prediction_Harshit_S.ipynb
├── requirements.txt
└── README.md
```

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/harshit-glitch24/Customer-Lifetime-Value-Prediction-Using-Statistical-Machine-Learning.git
cd Customer-Lifetime-Value-Prediction-Using-Statistical-Machine-Learning
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the notebook

```bash
jupyter lab
```

Open `Customer_Lifetime_Value_Prediction_Harshit_S.ipynb` and run the cells from top to bottom. Jupyter Notebook can be used instead of JupyterLab if preferred.

## Dataset schema

The generated dataset contains one row per customer.

| Feature | Description |
|---|---|
| `customer_id` | Unique customer identifier; excluded from model predictors |
| `age` | Customer age |
| `tenure_months` | Customer tenure in months |
| `orders_12m` | Number of orders in the historical 12-month window |
| `avg_order_value` | Average historical order value |
| `spend_12m` | Historical 12-month spend |
| `days_since_last_order` | Recency in days |
| `returns_rate` | Proportion of orders returned |
| `email_engagement` | Engagement score between 0 and 1 |
| `loyalty_member` | Loyalty-program membership |
| `acquisition_channel` | Customer acquisition channel |
| `region` | Broad customer region |
| `future_12m_value` | **Prediction target:** simulated spend in the following 12 months |

For a real-data implementation, replace the synthetic data-generation section with a controlled data-loading step and confirm that every predictor would be available at the actual prediction date. The target must represent a future observation window and must not leak information into the predictors.

## Reproducibility and responsible use

- A fixed random seed is used for repeatable synthetic-data generation and model splits.
- The notebook demonstrates missing-value handling and mixed-type feature processing.
- The target and customer identifier are excluded from model inputs.
- Synthetic data can encode artificial relationships that are easier for models to learn than real customer behavior.
- Before production use, validate temporal splits, drift, calibration, segment-level error, privacy requirements, and business impact on real data.

## Limitations and future improvements

- Replace synthetic records with an approved, representative customer dataset.
- Evaluate using a time-based holdout to better reflect future deployment.
- Compare against domain-appropriate CLV baselines and probabilistic customer-value models.
- Tune hyperparameters using nested or properly separated cross-validation.
- Add prediction intervals and segment-level error analysis.
- Track model drift and monitor business outcomes after deployment.

## Author

**Harshit S**  
Computer Science (Data Science) — Academic project  
GitHub: [@harshit-glitch24](https://github.com/harshit-glitch24)

---

If you use or extend this project, please preserve the synthetic-data caveat and document the dataset and evaluation setup used for any new results.
