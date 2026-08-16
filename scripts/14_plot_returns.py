import pandas as pd
import matplotlib.pyplot as plt

# Load daily returns data
df = pd.read_csv("data/processed/nifty_daily_returns.csv")

# Convert Date to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Plot daily returns
plt.figure(figsize=(12, 5))

plt.plot(df["Date"], df["Return"])

plt.title("NIFTY Daily Returns")
plt.xlabel("Date")
plt.ylabel("Daily Return (%)")

plt.axhline(y=0, linestyle="--", linewidth=1)

plt.tight_layout()
plt.show()