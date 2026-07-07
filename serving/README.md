# Model Serving API

Serves the best iris classifier (Logistic Regression, 96.67% — selected via MLflow experiments) over REST with FastAPI.

## Why FastAPI

A trained model is only useful if applications can call it. FastAPI is the industry standard for Python ML serving: async and fast, automatic input validation via Pydantic (bad requests get rejected with clear errors before reaching the model), and free interactive docs at `/docs`.

## Run

```bash
python save_model.py        # trains and saves model.joblib
uvicorn app:app --reload    # starts the API
```

Open http://127.0.0.1:8000/docs to try it in the browser, or:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

Response:

```json
{"species": "setosa", "confidence": 0.9784}
```

## Endpoints

- `GET /health` — liveness check (used by CI and load balancers)
- `POST /predict` — takes 4 measurements, returns species + confidence
