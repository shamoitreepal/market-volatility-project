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

# Estimate OLS
model = sm.OLS(y, X).fit()

# Display results
print(model.summary())

# Save regression results
with open(
    "output/results/volatility_ols_results.txt",
    "w"
) as f:
    f.write(model.summary().as_text())

print("\nOLS results saved successfully.")