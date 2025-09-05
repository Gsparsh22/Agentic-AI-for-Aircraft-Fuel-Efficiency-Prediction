# Aircraft Fuel Consumption Prediction (Random Forest)

This repository contains a ready-to-use project that **simulates aircraft flight data** and **trains a Random Forest regressor** to predict fuel consumption (kg/hour) from flight parameters.

---

## Project overview

The pipeline consists of three scripts:

- `data_simulation.py` — generate a synthetic dataset and save train/test CSV files (`data/train.csv`, `data/test.csv`, `data/metadata.json`).
- `train_model.py` — train a `RandomForestRegressor` using `GridSearchCV` (with cross-validation) and save the best model to `models/random_forest_fuel.pkl`.
- `evaluate_model.py` — load the saved model, evaluate on test data, print metrics (RMSE, MAE, R²), and save a predicted-vs-actual plot to `outputs/pred_vs_actual.png`.

All scripts are written for **Python 3.10+**, include docstrings and comments, and are reproducible via a fixed random seed.

---

## Files & folders

.
├── data_simulation.py
├── train_model.py
├── evaluate_model.py
├── data/ # created by data_simulation.py
│ ├── train.csv
│ ├── test.csv
│ └── metadata.json
├── models/ # created by train_model.py
│ └── random_forest_fuel.pkl
├── outputs/ # created by evaluate_model.py
│ └── pred_vs_actual.png
├── requirements.txt
└── README.md

---

## Requirements

- Python 3.10+
- Pip

Minimum packages (in `requirements.txt`):

```
numpy>=1.23
pandas>=1.5
scikit-learn>=1.1
matplotlib>=3.5
seaborn>=0.11
joblib>=1.1
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Quick start (copy & paste)

```bash
pip install -r requirements.txt
python data_simulation.py
python train_model.py
python evaluate_model.py
```

**What this does:**

- Generates a synthetic dataset (default 10,000 samples) and saves `data/train.csv` and `data/test.csv`.
- Runs `GridSearchCV` (moderate grid) to train a Random Forest and saves the best model to `models/random_forest_fuel.pkl`.
- Evaluates the saved model on the test set, prints RMSE/MAE/R², and saves a parity plot to `outputs/pred_vs_actual.png`.

---

## `data_simulation.py` — details & options

**Purpose:** Create a realistic-looking synthetic dataset for fuel consumption.

**Usage:**

```bash
python data_simulation.py
```

**Options:**

```bash
python data_simulation.py --n_samples 20000 --test_size 0.25 --seed 123
```

**Generated features:**

- speed (m/s)
- altitude (m)
- payload (kg)
- temperature (°C)
- wind_speed (m/s)
- air_density (kg/m³) — computed from altitude & temperature

**Target:**

- `fuel_consumption` (kg/hour) — computed from a simplified physics-based model (parasite + induced drag → power → fuel flow) with wind/temperature penalties and heteroscedastic noise.

---

## `train_model.py` — details & options

**Purpose:** Train a Random Forest regressor with CV and simple hyperparameter tuning.

**Usage:**

```bash
python train_model.py
```

**Options:**

- `--train_csv` — default: `data/train.csv`
- `--seed` — default: `42`

**Hyperparameter grid (default):**

- `n_estimators`: [100, 200]
- `max_depth`: [8, 12, None]
- `min_samples_split`: [2, 5]

**Outputs:**

- `models/random_forest_fuel.pkl`
- `models/training_summary.json`

*Tip:* If grid search is slow, reduce the grid or use fewer CV folds (`cv=3`).

---

## `evaluate_model.py` — details & options

**Purpose:** Load the trained model and evaluate on the hold-out test set.

**Usage:**

```bash
python evaluate_model.py
```

**Options:**

- `--model` — default: `models/random_forest_fuel.pkl`
- `--test_csv` — default: `data/test.csv`

**Metrics printed:**

- RMSE (kg/hr)
- MAE (kg/hr)
- R²

**Output file:**

- `outputs/pred_vs_actual.png` — predicted vs actual scatter plot (parity plot)

---

## Interpretation & notes

- **Units:** Target uses kg/hour. Approximate conversion to liters/hour (Jet-A):

```ini
liters = kg / 0.8   # (1 L ≈ 0.8 kg)
```

- **Why the model is realistic:** The synthetic target is generated from a toy but physics-inspired model (drag → power → fuel flow) with additive penalties and heteroscedastic noise. This creates non-linear relationships and variable noise — a good fit for tree-based models.

- **Reproducibility:** Default seed is 42. You can change it via script arguments.

---

## Troubleshooting & tips

- `GridSearchCV` may be CPU-intensive. Options:
  - Reduce `n_estimators` or the parameter grid.
  - Reduce `cv` folds (e.g., `cv=3`).
  - Set `n_jobs` to a fixed number if `-1` causes resource contention.

- On headless servers, use a non-interactive matplotlib backend before plotting:

```python
import matplotlib
matplotlib.use("Agg")
```

- If predictions are biased, verify train/test distribution overlap (`data/metadata.json`) and consider:
  - Feature engineering (e.g., `speed**2`, interactions),
  - Trying gradient-boosted models (XGBoost / LightGBM),
  - Adding domain priors or constraints.

---

## Next steps / extensions

- Add SHAP or permutation feature importance for model explainability.
- Try other regressors: XGBoost, LightGBM, or neural nets.
- Convert the target to liters/hour in the dataset and retrain.
- Create a Jupyter notebook walkthrough or Dockerfile for reproducible runs.

---

## Example terminal session (copy & paste)

```bash
# optional: create & activate a virtual environment
python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows (PowerShell)
venv\Scripts\Activate.ps1

# install dependencies
pip install -r requirements.txt

# run pipeline
python data_simulation.py
python train_model.py
python evaluate_model.py
```
