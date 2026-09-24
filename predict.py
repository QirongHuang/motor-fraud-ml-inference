import json
import pandas as pd
from xgboost import XGBClassifier

MODEL_PATH = "model/fraud_xgb_model.json"
FEATURE_PATH = "model/features.json"


# Load model
model = XGBClassifier()
model.load_model(MODEL_PATH)


# Load expected feature schema
with open(FEATURE_PATH, "r") as f:
    FEATURES = json.load(f)


def predict_fraud(data: pd.DataFrame) -> pd.DataFrame:
    """Generate fraud probabilities from prepared model features."""

    missing_features = [
        feature for feature in FEATURES
        if feature not in data.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    X = data[FEATURES].copy()

    probabilities = model.predict_proba(X)[:, 1]

    return pd.DataFrame({
        "fraud_probability": probabilities
    })


if __name__ == "__main__":
    sample = pd.read_csv("data/sample_inference.csv")

    predictions = predict_fraud(sample)
    print(f"Loaded {len(sample)} observations.")
    print(f"Using {len(FEATURES)} model features.")
    print("\nPredictions:")
    print(predictions)