from pathlib import Path

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# ==========================================================
# Project folders
# ==========================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

PROCESSED_DATA = PROJECT_DIR / "data" / "processed"
PLOTS = PROJECT_DIR / "plots"

# Create plots folder if it doesn't exist
PLOTS.mkdir(exist_ok=True)

# ==========================================================
# Load data
# ==========================================================

df = pd.read_csv(
    PROCESSED_DATA / "master_dataset_features.csv",
    parse_dates=["Date"]
)

print(df.head())
print(df.shape)

# ==========================================================
# Summary Statistics
# ==========================================================

summary = df.describe()

print("\nSummary Statistics\n")

print(summary)

summary.to_csv(
    PROCESSED_DATA / "summary_statistics.csv"
)

print("\nSummary statistics saved!")

# ==========================================================
# NIFTY Time Series
# ==========================================================

plt.figure(figsize=(12,6))

plt.plot(
    df["Date"],
    df["Nifty"]
)

plt.title("NIFTY Index (2015–2025)")

plt.xlabel("Year")

plt.ylabel("NIFTY")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    PLOTS / "nifty_timeseries.png",
    dpi=300
)

plt.show()

# ==========================================================
# Histogram of Monthly Returns
# ==========================================================

plt.figure(figsize=(8,5))

plt.hist(
    df["Nifty_Return"].dropna(),
    bins=20
)

plt.title("Distribution of Monthly NIFTY Returns")

plt.xlabel("Monthly Return (%)")

plt.ylabel("Frequency")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    PLOTS / "nifty_return_histogram.png",
    dpi=300
)

plt.show()


# ==========================================================
# Correlation Heatmap
# ==========================================================

import seaborn as sns

# Select only numeric columns
corr = df.select_dtypes(include=["number"]).corr()

plt.figure(figsize=(12, 8))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    center=0,
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Matrix")

plt.tight_layout()

plt.savefig(
    PLOTS / "03_correlation_heatmap.png",
    dpi=300
)

plt.show()


# ==========================================================
# Boxplots
# ==========================================================

variables = [
    "Nifty_Return",
    "Oil_Return",
    "FX_Return",
    "Inflation"
]

plt.figure(figsize=(12,6))

df[variables].boxplot()

plt.title("Boxplots of Monthly Returns and Inflation")

plt.ylabel("Value")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    PLOTS / "04_boxplots.png",
    dpi=300
)

plt.show()



# ==========================================================
# Oil Returns vs NIFTY Returns
# ==========================================================

plt.figure(figsize=(8,6))

plt.scatter(
    df["Oil_Return"],
    df["Nifty_Return"]
)

plt.xlabel("Oil Return (%)")

plt.ylabel("NIFTY Return (%)")

plt.title("Oil Returns vs NIFTY Returns")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    PLOTS / "05_oil_vs_nifty.png",
    dpi=300
)

plt.show()



# ==========================================================
# USDINR Returns vs NIFTY Returns
# ==========================================================

plt.figure(figsize=(8,6))

plt.scatter(
    df["FX_Return"],
    df["Nifty_Return"]
)

plt.xlabel("USDINR Return (%)")

plt.ylabel("NIFTY Return (%)")

plt.title("USDINR Returns vs NIFTY Returns")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    PLOTS / "06_fx_vs_nifty.png",
    dpi=300
)

plt.show()



# ==========================================================
# Inflation vs NIFTY Returns
# ==========================================================

plt.figure(figsize=(8,6))

plt.scatter(
    df["Inflation"],
    df["Nifty_Return"]
)

plt.xlabel("Inflation (%)")

plt.ylabel("NIFTY Return (%)")

plt.title("Inflation vs NIFTY Returns")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    PLOTS / "07_inflation_vs_nifty.png",
    dpi=300
)

plt.show()


# ==========================================================
# Rolling Volatility
# ==========================================================

df["Rolling_Volatility"] = (
    df["Nifty_Return"]
    .rolling(window=12)
    .std()
)

plt.figure(figsize=(12,6))

plt.plot(
    df["Date"],
    df["Rolling_Volatility"]
)

plt.title("12-Month Rolling Volatility")

plt.xlabel("Year")

plt.ylabel("Volatility")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    PLOTS / "08_rolling_volatility.png",
    dpi=300
)

plt.show()



