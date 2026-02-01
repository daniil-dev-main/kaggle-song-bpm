# Predicting Song Tempo (BPM)

This project builds a regression pipeline to estimate song tempo (BPM) from extracted audio features. 

Multiple models are evaluated, with XGBoost selected as the final model for inference on new data.

---

## Dataset

The project uses data provided by:
Walter Reade and Elizabeth Park (2025). Predicting the Beats-per-Minute of Songs. Kaggle Playground Series - Season 5, Episode 9. 
Available at: https://kaggle.com/competitions/playground-series-s5e9

Download the following files from the competition page and place them in the `data/` directory:

```
data/
├── train.csv
└── test.csv
```

- `train.csv` contains audio features and BPM targets.
- `test.csv` contains audio features only.

---

## Features

The model uses following audio features:

- `RhythmScore`
- `AudioLoudness`
- `Energy`
- `MoodScore`
- `VocalContent`
- `AcousticQuality`
- `InstrumentalScore`
- `LivePerformanceLikelihood`
- `TrackDurationM`


---

## Models

The following models are implemented and evaluated:

- Polynomial Linear Regression (scikit-learn)
- Linear Regression (matrix equations)
- **XGBoost Regressor** with randomized hyperparameter search

The final predictions are generated using the **best-performing XGBoost model**.

---

## Evaluation

Model performance is evaluated using **Root Mean Squared Error (RMSE)** on a validation set:

- 80% training data
- 20% validation data

RMSE is reported for each model to allow comparison.

---

## Project Structure

```
├── data/
│   ├── train.csv
│   └── test.csv
├── src/
│   ├── models/
│   │   └── *.joblib
│   ├── train.py
│   ├── predict.py
│   ├── features.py
│   ├── models.py
│   └── evaluate.py
├── submission/
│   └── output.csv
└── requirements.txt
```

---

## How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train models:
```bash
python src/train.py
```

3. Generate predictions and submission file:
```bash
python src/predict.py
```

The output file will be saved to:
```
submission/output.csv
```