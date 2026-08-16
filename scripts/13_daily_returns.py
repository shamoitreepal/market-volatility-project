import pandas as pd
import numpy as np

# Load daily NIFTY data
df = pd.read_csv("data/processed/nifty_clean.csv")

# Convert Date to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Sort data by date
df = df.sort_values("Date")

# Calculate daily log returns
df["Return"] = 100 * np.log(df["Close"] / df["Close"].shift(1))

# Remove the first missing observation
df = df.dropna()

# Display the first five observations
print(df.head())

# Display the last five observations
print(df.tail())

# Save the new dataset
df.to_csv("data/processed/nifty_daily_returns.csv", index=False)

print("\nDaily returns calculated successfully!")
print("Number of observations:", len(df))