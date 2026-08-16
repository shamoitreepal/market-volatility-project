from pathlib import Path
import pandas as pd
import numpy as np

# ==========================================================
# Project folders
# ==========================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

PROCESSED_DATA = PROJECT_DIR / "data" / "processed"

# ==========================================================
# Read master dataset
# ==========================================================

df = pd.read_csv(
    PROCESSED_DATA / "master_dataset.csv",
    parse_dates=["Date"]
)

print(df.head())
print(df.shape)


# ==========================================================
# Monthly NIFTY Returns (%)
# ==========================================================

df["Nifty_Return"] = df["Nifty"].pct_change() * 100

print(df[["Date", "Nifty", "Nifty_Return"]].head())


# ==========================================================
# Log Returns
# ==========================================================

df["Log_Return"] = np.log(
    df["Nifty"] / df["Nifty"].shift(1)
)

print(df[["Date", "Log_Return"]].head())


# ==========================================================
# Oil Price Return (%)
# ==========================================================

df["Oil_Return"] = df["Brent"].pct_change() * 100

# ==========================================================
# Exchange Rate Return (%)
# ==========================================================

df["FX_Return"] = df["USDINR"].pct_change() * 100

# ==========================================================
# Monthly Inflation (%)
# ==========================================================

df["Inflation"] = df["CPI"].pct_change() * 100

# ==========================================================
# Real Repo Rate
# ==========================================================

df["Real_Repo"] = df["Repo"] - df["Inflation"]

# ==========================================================
# Lagged Variables
# ==========================================================

df["Nifty_Lag1"] = df["Nifty_Return"].shift(1)

df["Oil_Lag1"] = df["Oil_Return"].shift(1)

df["FX_Lag1"] = df["FX_Return"].shift(1)

# ==========================================================
# Save feature engineered dataset
# ==========================================================

df.to_csv(
    PROCESSED_DATA / "master_dataset_features.csv",
    index=False
)

print("\nFeature Engineering Completed!\n")

print(df.head())

print("\nShape")
print(df.shape)

print("\nMissing Values")
print(df.isnull().sum())