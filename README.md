
## Projects

| # | Project | Type | Algorithm | Status |
|---|---------|------|-----------|--------|
| 1 | [Iris Species Classifier](./classification) | Classification | Random Forest |  
| 2 | [Diabetes Progression Predictor](./regression) | Regression | Linear Regression, Random Forest | 
| 3 | MLOps Pipeline with MLflow (./mlops-mlflow) | Experiment tracking & model registry | — | 🔜 Planned |

## Repo structure

```
.
├── classification/    # Project 1: iris species prediction
│   ├── train.py
│   └── iris.csv
├── regression/        # Project 2: diabetes progression prediction
│   ├── train.py
│   └── diabetes.csv
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
cd classification && python train.py   # or cd regression
```

---

## Algorithms used

### Random Forest (classification & regression)

 An ensemble method that builds many decision trees, each trained on a random subset of the data and features, and combines their outputs — majority vote for classification, average for regression.

It works well out of the box with almost no tuning, handles non-linear relationships, is resistant to overfitting (individual trees overfit, but averaging cancels that out), and gives feature-importance scores for free.

Where it's used
In the real world Credit risk scoring in banking, disease diagnosis from patient data, fraud detection, and customer churn prediction — anywhere you have tabular data and want strong accuracy without heavy tuning.

**Result:** 90% accuracy on the iris test set (30 samples, 3 species).

### Linear Regression

 The simplest regression algorithm — fits a straight line (or hyperplane) that minimizes the squared difference between predicted and actual values. Each feature gets a coefficient showing how much it pushes the prediction up or down.

Why I used it It's the standard baseline for any regression problem. It's fast, fully interpretable (you can read the coefficients like a recipe), and if a complex model can't beat it, the extra complexity isn't worth it. On the diabetes dataset it actually edges out Random Forest — proof that simple models often win on small datasets.

Where it's used 
In the real world House price estimation, sales forecasting, medical risk scoring, and economics — anywhere the relationship between inputs and output is roughly linear and interpretability matters.

Result:  R² 0.45, RMSE ~54 on the diabetes test set (vs R² 0.44 for Random Forest).


MLOPS 
An open-source platform for the ML lifecycle: experiment tracking (log every run's parameters, metrics, and artifacts), a model registry (versioned, production-ready models), and model serving.

Training models is easy; managing dozens of experiments is not. MLflow gives a searchable history of every run and versions the best model — the foundation of taking ML to production.

Where it's used in the real world: Industry-standard at companies running many models in production; integrated into Databricks, Azure ML, and most modern ML platforms.

Result: 
I tracked experiments; best model (Logistic Regression, 96.67%) registered as `iris-classifier` v1
