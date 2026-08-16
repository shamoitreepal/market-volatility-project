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

# ==========================================================
# Model 1
# ==========================================================

m1 = df[
    [
        "Nifty_Return",
        "Oil_Return",
        "FX_Return",
        "Inflation",
        "Real_Repo"
    ]
].dropna()

y1 = m1["Nifty_Return"]

X1 = sm.add_constant(
    m1[
        [
            "Oil_Return",
            "FX_Return",
            "Inflation",
            "Real_Repo"
        ]
    ]
)

model1 = sm.OLS(y1, X1).fit()

# ==========================================================
# Model 2
# ==========================================================

m2 = df[
    [
        "Nifty_Return",
        "Oil_Lag1",
        "FX_Lag1",
        "Inflation",
        "Real_Repo"
    ]
].dropna()

y2 = m2["Nifty_Return"]

X2 = sm.add_constant(
    m2[
        [
            "Oil_Lag1",
            "FX_Lag1",
            "Inflation",
            "Real_Repo"
        ]
    ]
)

model2 = sm.OLS(y2, X2).fit()

# ==========================================================
# Model 3
# ==========================================================

df["COVID_Dummy"] = (
    df["Date"] >= "2020-03-01"
).astype(int)

m3 = df[
    [
        "Nifty_Return",
        "Oil_Return",
        "FX_Return",
        "Inflation",
        "Real_Repo",
        "COVID_Dummy"
    ]
].dropna()

y3 = m3["Nifty_Return"]

X3 = sm.add_constant(
    m3[
        [
            "Oil_Return",
            "FX_Return",
            "Inflation",
            "Real_Repo",
            "COVID_Dummy"
        ]
    ]
)

model3 = sm.OLS(y3, X3).fit()

# ==========================================================
# Build Comparison Table
# ==========================================================

comparison = pd.DataFrame({

    "Model 1":
    model1.params,

    "Model 2":
    model2.params,

    "Model 3":
    model3.params

})

comparison.loc["R_squared"] = [
    model1.rsquared,
    model2.rsquared,
    model3.rsquared
]

comparison.loc["Adj_R_squared"] = [
    model1.rsquared_adj,
    model2.rsquared_adj,
    model3.rsquared_adj
]

comparison.loc["AIC"] = [
    model1.aic,
    model2.aic,
    model3.aic
]

comparison.loc["BIC"] = [
    model1.bic,
    model2.bic,
    model3.bic
]

print(comparison)

comparison.to_csv(
    PROCESSED_DATA / "model_comparison.csv"
)

print("\nModel comparison saved successfully!")