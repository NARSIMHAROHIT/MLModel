# Iris Classification Model

A simple ML classification model that predicts iris flower species (setosa, versicolor, virginica) from 4 measurements, using a Random Forest classifier.

## Files

- `iris.csv` — dataset (150 samples, 4 features + species label)
- `train.py` — loads data, trains the model, prints accuracy
- `requirements.txt` — dependencies

## Setup & Run

```bash
pip install -r requirements.txt
python train.py
```

## Expected output

~90% accuracy on the test set, plus a per-class classification report and one example prediction.
