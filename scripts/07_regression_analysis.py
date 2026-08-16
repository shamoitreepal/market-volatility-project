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
# Load dataset
# ==========================================================

df = pd.read_csv(
    PROCESSED_DATA / "master_dataset_features.csv",
    parse_dates=["Date"]
)

print("Dataset Loaded!")
print(df.shape)

# ==========================================================
# Variables
# ==========================================================

variables = [
    "Nifty_Return",
    "Oil_Return",
    "FX_Return",
    "Inflation",
    "Real_Repo"
]

data = df[variables].dropna()

print("\nRegression Dataset Shape")
print(data.shape)

# ==========================================================
# Dependent Variable
# ==========================================================

y = data["Nifty_Return"]

# ==========================================================
# Independent Variables
# ==========================================================

X = data[
    [
        "Oil_Return",
        "FX_Return",
        "Inflation",
        "Real_Repo"
    ]
]

# Add intercept

X = sm.add_constant(X)

# ==========================================================
# Estimate OLS
# ==========================================================

model = sm.OLS(y, X).fit()

# ==========================================================
# Results
# ==========================================================

print(model.summary())