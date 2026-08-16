import pandas as pd
import statsmodels.api as sm

# Load merged dataset
df = pd.read_csv(
    "data/processed/master_dataset_volatility.csv"
)

# Select variables
variables = [
    "Realized_Volatility",
    "Oil_Return",
    "FX_Return",
    "Inflation",
    "Real_Repo"
]

# Remove missing observations
data = df[variables].dropna()

# Dependent variable
y = data["Realized_Volatility"]

# Independent variables
X = data[
    [
        "Oil_Return",
        "FX_Return",
        "Inflation",
        "Real_Repo"
    ]
]

# Add constant
X = sm.add_constant(X)

# Estimate OLS with HAC standard errors
model = sm.OLS(y, X).fit(
    cov_type="HAC",
    cov_kwds={"maxlags": 1}
)

# Display results
print(model.summary())

# Save results
with open(
    "output/results/volatility_ols_hac_results.txt",
    "w"
) as f:
    f.write(model.summary().as_text())

print("\nHAC regression results saved successfully.")