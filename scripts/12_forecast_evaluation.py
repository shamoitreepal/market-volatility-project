from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================================
# Project folders
# ==========================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

PROCESSED_DATA = PROJECT_DIR / "data" / "processed"
PLOTS = PROJECT_DIR / "plot"

PLOTS.mkdir(exist_ok=True)

# ==========================================================
# Load dataset
# ==========================================================

df = pd.read_csv(
    PROCESSED_DATA / "master_dataset_features.csv",
    parse_dates=["Date"]
)

# ==========================================================
# Regression Dataset
# ==========================================================

variables = [
    "Date",
    "Nifty_Return",
    "Oil_Return",
    "FX_Return",
    "Inflation",
    "Real_Repo"
]

df = df[variables].dropna()

# ==========================================================
# Build Model
# ==========================================================

y = df["Nifty_Return"]

X = df[
    [
        "Oil_Return",
        "FX_Return",
        "Inflation",
        "Real_Repo"
    ]
]

X = sm.add_constant(X)

model = sm.OLS(y, X).fit()

# ==========================================================
# Predictions
# ==========================================================

df["Predicted"] = model.predict(X)

df["Residual"] = df["Nifty_Return"] - df["Predicted"]

# ==========================================================
# Accuracy Measures
# ==========================================================

rmse = mean_squared_error(
    df["Nifty_Return"],
    df["Predicted"]
) ** 0.5

mae = mean_absolute_error(
    df["Nifty_Return"],
    df["Predicted"]
)

r2 = r2_score(
    df["Nifty_Return"],
    df["Predicted"]
)

print("="*60)
print("Forecast Accuracy")
print("="*60)

print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R²   : {r2:.4f}")

# ==========================================================
# Actual vs Predicted
# ==========================================================

plt.figure(figsize=(12,6))

plt.plot(
    df["Date"],
    df["Nifty_Return"],
    label="Actual"
)

plt.plot(
    df["Date"],
    df["Predicted"],
    label="Predicted"
)

plt.title("Actual vs Predicted Monthly NIFTY Returns")

plt.xlabel("Date")

plt.ylabel("Monthly Return (%)")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    PLOTS / "actual_vs_predicted.png",
    dpi=300
)

plt.show()

# ==========================================================
# Residual Plot
# ==========================================================

plt.figure(figsize=(12,5))

plt.plot(
    df["Date"],
    df["Residual"]
)

plt.axhline(
    0,
    linestyle="--"
)

plt.title("Regression Residuals")

plt.xlabel("Date")

plt.ylabel("Residual")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    PLOTS / "residual_plot.png",
    dpi=300
)

plt.show()

print("\nForecast evaluation completed successfully!")