"""
This script loads the trained model, applies it to the test dataset,
and saves predictions in the required submission format.
"""

import pandas as pd
from pathlib import Path
from joblib import load

from features import prepare_features

def main() -> None:
    """
    Runs the inference pipeline and generates a submission file.
    """
    # Define file paths
    root = Path(__file__).resolve().parents[1]
    test_path = root / "data" / "test.csv"
    out_path = root / "submission" / "output.csv"

    # Load test data and prepare features
    test_df = pd.read_csv(test_path)
    x_test = prepare_features(test_df)

    # Load trained model and generate predictions
    model = load("models/xgb_best.joblib")
    predicted_val = model.predict(x_test)

    # Create submission DataFrame and save it
    submission = pd.DataFrame({
        "id": test_df["id"],
        "BeatsPerMinute": predicted_val
    })
    submission.to_csv(out_path, index=False)
    print(f"Saved submission to: {out_path}")

if __name__ == "__main__":
    main()