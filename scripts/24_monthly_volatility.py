import pandas as pd
import numpy as np

# Load daily NIFTY returns
df = pd.read_csv("data/processed/nifty_daily_returns.csv")

# Convert Date to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Create month identifier
df["Month"] = df["Date"].dt.to_period("M")

# Calculate monthly actual volatility
monthly_volatility = (
    df.groupby("Month")["Return"]
    .apply(lambda x: np.sqrt(np.sum(x ** 2)))
    .reset_index()
)

# Convert Month back to date
monthly_volatility["Date"] = monthly_volatility["Month"].dt.to_timestamp("M")

# Remove Month column
monthly_volatility = monthly_volatility.drop(columns=["Month"])

# Rename volatility column
monthly_volatility = monthly_volatility.rename(
    columns={"Return": "Realized_Volatility"}
)

# Display results
print("\nMONTHLY REALIZED VOLATILITY")
print("===========================")
print(monthly_volatility.head())
print(monthly_volatility.tail())

print("\nNumber of months:", len(monthly_volatility))

# Save dataset
monthly_volatility.to_csv(
    "data/processed/nifty_monthly_volatility.csv",
    index=False
)

print("\nMonthly volatility dataset saved successfully.")

import os

print(os.listdir("data/processed"))