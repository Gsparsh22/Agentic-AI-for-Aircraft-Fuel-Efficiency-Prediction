#!/usr/bin/env python3
# train_model.py
"""
Train a Random Forest regressor on the simulated aircraft fuel-consumption dataset.

- Loads data/data_train.csv and data/test.csv (or data/train.csv)
- Performs hyperparameter tuning (GridSearchCV) with cross-validation
- Saves best trained model to models/random_forest_fuel.pkl
- Saves training summary (best params and CV results) to models/training_summary.json

Usage:
    python train_model.py
"""
from __future__ import annotations

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

RNG_SEED = 42


def load_data(train_path: str = "data/train.csv") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load training data from CSV.

    Args:
        train_path: path to train CSV

    Returns:
        X (DataFrame), y (Series)
    """
    df = pd.read_csv(train_path)
    feature_cols = ["speed", "altitude", "payload", "temperature", "wind_speed", "air_density"]
    X = df[feature_cols].copy()
    y = df["fuel_consumption"].copy()
    return X, y


def train_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int = RNG_SEED,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train a RandomForestRegressor with GridSearchCV hyperparameter tuning.

    Returns:
        best_estimator, summary_dict
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV

    # Model with random_state for reproducibility
    base_model = RandomForestRegressor(random_state=seed)

    # Hyperparameter grid (kept moderate so this runs in a reasonable time)
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [8, 12, None],
        "min_samples_split": [2, 5],
    }

    # Grid search with 5-fold CV, scoring using negative RMSE
    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=5,
        n_jobs=-1,
        verbose=2,
        return_train_score=True,
    )

    print("[INFO] Starting GridSearchCV...")
    grid.fit(X, y)
    print("[INFO] Grid search complete.")
    best = grid.best_estimator_

    # Summarize CV results (best params, best score)
    summary = {
        "best_params": grid.best_params_,
        "best_cv_score_neg_rmse": float(grid.best_score_),
        "cv_results_keys": list(grid.cv_results_.keys()),
    }

    return best, summary


def save_model(model: Any, summary: Dict[str, Any], output_path: str = "models/random_forest_fuel.pkl"):
    """
    Save the trained model (joblib) and a JSON summary.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    summary_path = os.path.join(os.path.dirname(output_path), "training_summary.json")
    with open(summary_path, "w", encoding="utf8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[INFO] Saved model -> {output_path}")
    print(f"[INFO] Saved summary -> {summary_path}")


def main():
    """Main entrypoint for training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train Random Forest model for fuel consumption prediction")
    parser.add_argument("--train_csv", type=str, default="data/train.csv", help="Path to training CSV")
    parser.add_argument("--seed", type=int, default=RNG_SEED, help="Random seed")
    args = parser.parse_args()

    X_train, y_train = load_data(args.train_csv)
    print(f"[INFO] Loaded training data: X={X_train.shape}, y={y_train.shape}")

    model, summary = train_random_forest(X_train, y_train, seed=args.seed)
    save_model(model, summary, output_path="models/random_forest_fuel.pkl")
    print("[INFO] Training complete.")


if __name__ == "__main__":
    main()
