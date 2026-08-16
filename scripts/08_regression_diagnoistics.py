from pathlib import Path

import pandas as pd
import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.diagnostic import acorr_breusch_godfrey

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

variables = [
    "Nifty_Return",
    "Oil_Return",
    "FX_Return",
    "Inflation",
    "Real_Repo"
]

df = df[variables].dropna()

# ==========================================================
# Regression
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
# VIF
# ==========================================================

print("\n")
print("="*60)
print("Variance Inflation Factor (VIF)")
print("="*60)

vif = pd.DataFrame()

vif["Variable"] = X.columns

vif["VIF"] = [
    variance_inflation_factor(X.values, i)
    for i in range(X.shape[1])
]

print(vif)

# ==========================================================
# Breusch-Pagan Test
# ==========================================================

print("\n")
print("="*60)
print("Breusch-Pagan Test")
print("="*60)

bp = het_breuschpagan(
    model.resid,
    model.model.exog
)

labels = [
    "LM Statistic",
    "LM p-value",
    "F Statistic",
    "F p-value"
]

for label, value in zip(labels, bp):
    print(label, ":", value)

# ==========================================================
# Breusch-Godfrey Test
# ==========================================================

print("\n")
print("="*60)
print("Breusch-Godfrey Test")
print("="*60)

bg = acorr_breusch_godfrey(
    model,
    nlags=2
)

labels = [
    "LM Statistic",
    "LM p-value",
    "F Statistic",
    "F p-value"
]

for label, value in zip(labels, bg):
    print(label, ":", value)

# ==========================================================
# Robust Standard Errors
# ==========================================================

print("\n")
print("="*60)
print("OLS with HC3 Robust Standard Errors")
print("="*60)

robust_model = model.get_robustcov_results(
    cov_type="HC3"
)

print(robust_model.summary())