import pandas as pd

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

# Calculate correlation matrix
correlation = df[variables].corr()

# Display correlation with NIFTY volatility
print("\nCORRELATION WITH NIFTY VOLATILITY")
print("------------------------------------")

print(
    correlation["Realized_Volatility"]
    .sort_values(ascending=False)
)

# Save correlation matrix
correlation.to_csv(
    "output/results/volatility_correlations.csv"
)

print("\nCorrelation matrix saved successfully.")