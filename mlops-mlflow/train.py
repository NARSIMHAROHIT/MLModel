"""MLOps with MLflow: experiment tracking + model registry.

Trains 3 classifiers on the iris dataset. Every run is logged to MLflow
(parameters, metrics, and the trained model artifact). The best model by
test accuracy is registered in the MLflow Model Registry.

Run it, then inspect results in the UI:
    python train.py
    mlflow ui --backend-store-uri sqlite:///mlflow.db
    # open http://127.0.0.1:5000
"""

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

# --- MLflow setup: sqlite backend so the Model Registry works locally ---
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("iris-classification")

# --- Data ---
df = pd.read_csv("../classification/iris.csv")
X = df.drop("species", axis=1)
y = df["species"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Candidate models ---
candidates = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
}

best = {"accuracy": 0.0, "run_id": None, "name": None}

for name, model in candidates.items():
    with mlflow.start_run(run_name=name) as run:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        # Log everything about this experiment
        mlflow.log_params(model.get_params())
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1)
        mlflow.sklearn.log_model(model, name="model")

        print(f"{name:22s}  accuracy: {acc:.2%}  f1: {f1:.3f}")

        if acc > best["accuracy"]:
            best = {"accuracy": acc, "run_id": run.info.run_id, "name": name}

# --- Register the winner in the Model Registry ---
model_uri = f"runs:/{best['run_id']}/model"
result = mlflow.register_model(model_uri, "iris-classifier")
print(f"\nBest model: {best['name']} ({best['accuracy']:.2%})")
print(f"Registered as 'iris-classifier' version {result.version}")
