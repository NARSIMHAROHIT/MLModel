# MLOps with MLflow

Experiment tracking and model registry for the iris classification problem.

## What this does

1. Trains 3 candidate models (Logistic Regression, Random Forest, Gradient Boosting)
2. Logs every run to MLflow: hyperparameters, accuracy, F1 score, and the model artifact itself
3. Registers the best model in the MLflow Model Registry as `iris-classifier`

## Why MLflow

In real ML work you train dozens of model variants. Without tracking, you lose track of which settings produced which results. MLflow gives you a searchable history of every experiment, and the Model Registry versions your production-ready models — the foundation of MLOps.

## Run

```bash
python train.py
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open http://127.0.0.1:5000 to compare runs side by side, view metrics, and inspect the registered model.

## Results

| Model | Accuracy | F1 (macro) |
|-------|----------|------------|
| Logistic Regression | 96.67% | 0.967 |
| Gradient Boosting | 96.67% | 0.967 |
| Random Forest | 90.00% | 0.900 |

Best model (`logistic_regression`) registered as `iris-classifier` v1.
