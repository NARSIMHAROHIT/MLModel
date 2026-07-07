"""Model serving API: exposes the trained iris classifier over REST.

Run:
    python save_model.py          # once, to create model.joblib
    uvicorn app:app --reload      # then open http://127.0.0.1:8000/docs
"""

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Iris Classifier API", version="1.0")
model = joblib.load("model.joblib")

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


class IrisInput(BaseModel):
    sepal_length: float = Field(..., example=5.1, gt=0)
    sepal_width: float = Field(..., example=3.5, gt=0)
    petal_length: float = Field(..., example=1.4, gt=0)
    petal_width: float = Field(..., example=0.2, gt=0)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: IrisInput):
    X = pd.DataFrame([[getattr(data, f) for f in FEATURES]], columns=FEATURES)
    species = model.predict(X)[0]
    probs = model.predict_proba(X)[0]
    confidence = float(max(probs))
    return {"species": species, "confidence": round(confidence, 4)}
