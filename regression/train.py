"""Regression model: predicts diabetes disease progression from patient features.

Compares a Linear Regression baseline against a Random Forest Regressor.
Dataset: 442 patients, 10 features (age, bmi, blood pressure, blood serum
measurements). Target: a quantitative measure of disease progression one
year after baseline.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load dataset
df = pd.read_csv("diabetes.csv")
X = df.drop("progression", axis=1)  # features: age, bmi, bp, serum levels...
y = df["progression"]               # target: disease progression score

# 2. Split into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train and evaluate both models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)
    print(f"{name:18s}  RMSE: {rmse:.2f}  R²: {r2:.3f}")

# 4. Predict on one test sample
sample = X_test.iloc[[0]]
pred = models["Linear Regression"].predict(sample)[0]
print(f"\nSample prediction: {pred:.1f} (actual: {y_test.iloc[0]:.1f})")
