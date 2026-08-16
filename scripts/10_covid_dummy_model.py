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

print("Dataset Loaded Successfully!")
print(df.shape)

# ==========================================================
# Create COVID Dummy
# ==========================================================

df["COVID_Dummy"] = (
    df["Date"] >= "2020-03-01"
).astype(int)

print("\nCOVID Dummy Counts")
print(df["COVID_Dummy"].value_counts())

# ==========================================================
# Keep Required Variables
# ==========================================================

variables = [
    "Nifty_Return",
    "Oil_Return",
    "FX_Return",
    "Inflation",
    "Real_Repo",
    "COVID_Dummy"
]

df = df[variables].dropna()

print("\nRegression Dataset Shape")
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
        "Oil_Return",
        "FX_Return",
        "Inflation",
        "Real_Repo",
        "COVID_Dummy"
    ]
]

X = sm.add_constant(X)

# ==========================================================
# Estimate OLS
# ==========================================================

model = sm.OLS(y, X).fit()

print(model.summary())