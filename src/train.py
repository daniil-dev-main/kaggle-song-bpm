"""
Training and evaluation pipeline for regression models.

This script loads training data, splits it into training and validation
sets, trains multiple regression models, evaluates them using RMSE,
and saves the trained models to disk.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump

from features import make_xy
from models import skl_poly_lr, lin_reg, xgb_random_search
from evaluate import rmse

def main() -> None:
    """
        Runs the full training and evaluation pipeline.
    """
    # Define paths
    root = Path(__file__).resolve().parents[1]
    train_path = root / "data" / "train.csv"
    model_path = Path("models")
    model_path.mkdir(parents=True, exist_ok=True)

    # Load dataset and construct features and targets
    bpm_training_data = pd.read_csv(train_path)
    x, y = make_xy(bpm_training_data)

    # Split data into training and validation sets
    train_x, val_x, train_y, val_y = train_test_split(x, y, test_size=0.2, random_state=1)

    # ---------------- Sklearn polynomial linear regression ----------------
    model_1, val_x_transformed = skl_poly_lr(train_x, train_y, val_x)
    dump(model_1, model_path / "skl_poly_lr.joblib")
    predicted_m1 = model_1.predict(val_x_transformed)

    # ---------------- Linear regression ----------------
    model_2 = lin_reg(train_x, train_y)
    dump(model_2, model_path / "lin_reg.joblib")
    predicted_m2 = val_x.values @ model_2

    # -------------- XGBoost with randomized hyperparameter search -------------
    xgb_best = xgb_random_search(train_x, train_y, n_iter=20)
    dump(xgb_best, model_path / "xgb_best.joblib")
    predicted_y_xgb = xgb_best.predict(val_x)

    # ---------------- Model evaluation ----------------
    print("RMSE of SKLearn linear regression model:", rmse(val_y, predicted_m1))
    print("RMSE of linear regression model:", rmse(val_y, predicted_m2))
    print("RMSE of XGBoost with best parameters:", rmse(val_y, predicted_y_xgb))

if __name__ == "__main__":
    main()