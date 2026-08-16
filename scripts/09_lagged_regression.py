from pathlib import Path

import pandas as pd
import statsmodels.api as sm

# ==========================================================
# Project folders
# ==========================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

PROCESSED_DATA = PROJECT_DIR / "data" / "processed"

# ==========================================================
# Load data
# ==========================================================

df = pd.read_csv(
    PROCESSED_DATA / "master_dataset_features.csv",
    parse_dates=["Date"]
)

print("Dataset Loaded")
print(df.shape)

# ==========================================================
# Keep required variables
# ==========================================================

variables = [
    "Nifty_Return",
    "Oil_Lag1",
    "FX_Lag1",
    "Inflation",
    "Real_Repo"
]

df = df[variables].dropna()

print("\nRegression Dataset")
print(df.shape)

# ==========================================================
# Dependent Variable
# ==========================================================

y = df["Nifty_Return"]

# ==========================================================
# Independent Variables
# ==========================================================

X = df[
    [
        "Oil_Lag1",
        "FX_Lag1",
        "Inflation",
        "Real_Repo"
    ]
]

X = sm.add_constant(X)

# ==========================================================
# Estimate Model
# ==========================================================

model = sm.OLS(y, X).fit()

print(model.summary())