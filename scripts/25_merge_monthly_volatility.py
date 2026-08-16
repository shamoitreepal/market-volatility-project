import pandas as pd

# Load monthly volatility
volatility = pd.read_csv(
    "data/processed/nifty_monthly_volatility.csv"
)

# Load existing macro dataset
macro = pd.read_csv(
    "data/processed/master_dataset_features.csv"
)

# Convert Date columns to datetime
volatility["Date"] = pd.to_datetime(volatility["Date"])
macro["Date"] = pd.to_datetime(macro["Date"])

# Merge the two datasets
merged = pd.merge(
    macro,
    volatility,
    on="Date",
    how="inner"
)

# Sort by date
merged = merged.sort_values("Date")

# Display information
print("\nMERGED DATASET")
print("==============")

print("\nFirst 5 observations:")
print(merged.head())

print("\nLast 5 observations:")
print(merged.tail())

print("\nNumber of observations:", len(merged))

print("\nVariables:")
print(merged.columns.tolist())

# Check missing values
print("\nMissing values:")
print(merged.isnull().sum())

# Save merged dataset
merged.to_csv(
    "data/processed/master_dataset_volatility.csv",
    index=False
)

print("\nMerged dataset saved successfully.")