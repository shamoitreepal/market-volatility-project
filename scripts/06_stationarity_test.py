from pathlib import Path

import pandas as pd

from statsmodels.tsa.stattools import adfuller

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
# Function for ADF Test
# ==========================================================

def adf_test(series, name):

    series = series.dropna()

    result = adfuller(series)

    print("\n" + "=" * 60)
    print(f"ADF Test : {name}")
    print("=" * 60)

    print(f"ADF Statistic : {result[0]:.4f}")
    print(f"P-value       : {result[1]:.4f}")
    print(f"Lags Used     : {result[2]}")
    print(f"Observations  : {result[3]}")

    print("\nCritical Values")

    for key, value in result[4].items():
        print(f"{key} : {value:.4f}")

    if result[1] < 0.05:
        print("\nResult : Stationary")
    else:
        print("\nResult : Non-Stationary")

# ==========================================================
# Variables to Test
# ==========================================================

variables = [
    "Nifty",
    "Nifty_Return",
    "USDINR",
    "FX_Return",
    "Brent",
    "Oil_Return",
    "CPI",
    "Inflation",
    "Repo",
    "Real_Repo"
]

for variable in variables:
    adf_test(df[variable], variable)

print("\nAll stationarity tests completed!")

