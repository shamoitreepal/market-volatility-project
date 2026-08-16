import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

# Load data
df = pd.read_csv(
    "data/processed/master_dataset_volatility.csv"
)

# Create lagged volatility
df["Volatility_Lag1"] = df["Realized_Volatility"].shift(1)

# Select independent variables
variables = [
    "Volatility_Lag1",
    "Oil_Return",
    "FX_Return",
    "Inflation",
    "Real_Repo"
]

# Remove missing observations
data = df[variables].dropna()

# Add constant
X = sm.add_constant(data)

# Calculate VIF
vif = pd.DataFrame()

vif["Variable"] = X.columns

vif["VIF"] = [
    variance_inflation_factor(X.values, i)
    for i in range(X.shape[1])
]

print("\nVIF RESULTS")
print("---------------")
print(vif)

# Save results
vif.to_csv(
    "output/results/volatility_vif.csv",
    index=False
)

print("\nVIF results saved successfully.")