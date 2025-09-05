#!/usr/bin/env python3
# evaluate_model.py
"""
Evaluate the trained Random Forest model on the test set and plot predictions.

- Loads models/random_forest_fuel.pkl
- Loads data/test.csv
- Computes RMSE, MAE, R^2
- Plots predicted vs actual fuel consumption and saves to outputs/pred_vs_actual.png

Usage:
    python evaluate_model.py
"""
from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple

RNG_SEED = 42


def load_test_data(test_path: str = "data/test.csv") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load test dataset.
    """
    df = pd.read_csv(test_path)
    feature_cols = ["speed", "altitude", "payload", "temperature", "wind_speed", "air_density"]
    X_test = df[feature_cols].copy()
    y_test = df["fuel_consumption"].copy()
    return X_test, y_test


def evaluate_model(model_path: str = "models/random_forest_fuel.pkl", test_csv: str = "data/test.csv"):
    """
    Load model and test data, compute metrics, and produce a scatter plot of predictions vs actuals.
    """
    os.makedirs("outputs", exist_ok=True)

    print(f"[INFO] Loading model from {model_path} ...")
    model = joblib.load(model_path)

    X_test, y_test = load_test_data(test_csv)
    print(f"[INFO] Loaded test data: X={X_test.shape}, y={y_test.shape}")

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "RMSE (kg/hr)": float(rmse),
        "MAE  (kg/hr)": float(mae),
        "R2   ": float(r2),
    }

    print("[RESULTS] Evaluation metrics on test set:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Plot predicted vs actual
    plt.figure(figsize=(8, 8))
    sns.set(style="whitegrid")
    ax = sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, s=20)
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    # Perfect prediction line
    plt.plot(lims, lims, "--", color="k", linewidth=1)
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel("Actual fuel consumption (kg/hr)")
    plt.ylabel("Predicted fuel consumption (kg/hr)")
    plt.title("Predicted vs Actual Fuel Consumption")
    plt.tight_layout()
    out_path = os.path.join("outputs", "pred_vs_actual.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[INFO] Saved plot -> {out_path}")


def main():
    """Main entrypoint."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate trained Random Forest model on test set")
    parser.add_argument("--model", type=str, default="models/random_forest_fuel.pkl", help="Path to model .pkl")
    parser.add_argument("--test_csv", type=str, default="data/test.csv", help="Path to test CSV")
    args = parser.parse_args()

    evaluate_model(model_path=args.model, test_csv=args.test_csv)


if __name__ == "__main__":
    main()
