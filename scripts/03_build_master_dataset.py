from pathlib import Path
import pandas as pd

# ==========================================================
# Project folders
# ==========================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

RAW_DATA = PROJECT_DIR / "data" / "raw"
PROCESSED_DATA = PROJECT_DIR / "data" / "processed"

# ==========================================================
# Read monthly market datasets
# ==========================================================

nifty = pd.read_csv(
    PROCESSED_DATA / "nifty_monthly.csv",
    parse_dates=["Date"]
)

usdinr = pd.read_csv(
    PROCESSED_DATA / "usdinr_monthly.csv",
    parse_dates=["Date"]
)

brent = pd.read_csv(
    PROCESSED_DATA / "brent_monthly.csv",
    parse_dates=["Date"]
)

# Rename columns

nifty.rename(columns={"Close": "Nifty"}, inplace=True)
usdinr.rename(columns={"Close": "USDINR"}, inplace=True)
brent.rename(columns={"Close": "Brent"}, inplace=True)

print("Market datasets loaded successfully!\n")

print(nifty.head())
print(usdinr.head())
print(brent.head())



# ==========================================================
# Read CPI dataset
# ==========================================================

cpi = pd.read_excel(
    RAW_DATA / "cpi_monthly.xlsx"
)

print("\nOriginal CPI Dataset")
print(cpi.head())

# ==========================================================
# Keep only required columns
# ==========================================================

cpi = cpi[[
    "year",
    "month",
    "index"
]]

# Rename

cpi.rename(
    columns={
        "index": "CPI"
    },
    inplace=True
)

print("\nCleaned CPI")
print(cpi.head())



# ==========================================================
# Create Date column
# ==========================================================

# Combine year and month into a date
cpi["Date"] = pd.to_datetime(
    cpi["year"].astype(str) + "-" + cpi["month"],
    format="%Y-%B"
)

# Move the date to the last day of the month
cpi["Date"] = cpi["Date"] + pd.offsets.MonthEnd(0)

# Keep only Date and CPI
cpi = cpi[["Date", "CPI"]]

# Sort from oldest to newest
cpi = cpi.sort_values("Date").reset_index(drop=True)

print("\nProcessed CPI")
print(cpi.head())
print(cpi.tail())
print(cpi.shape)


# ==========================================================
# Read Repo Rate
# ==========================================================

repo = pd.read_excel(
    RAW_DATA / "repo_rate.xlsx"
)

print("\nOriginal Repo Dataset")
print(repo.head())

# ==========================================================
# Rename columns (optional if already correct)
# ==========================================================

repo.rename(
    columns={
        "Repo_Rate": "Repo"
    },
    inplace=True
)

# ==========================================================
# Convert Month column to datetime
# ==========================================================

repo["Month"] = pd.to_datetime(repo["Month"])

# ==========================================================
# Move each observation to month-end
# ==========================================================

repo["Date"] = repo["Month"] + pd.offsets.MonthEnd(0)

# ==========================================================
# Keep only required columns
# ==========================================================

repo = repo[["Date", "Repo"]]

# ==========================================================
# If multiple policy changes occur in one month,
# keep the last one
# ==========================================================

repo = (
    repo
    .sort_values("Date")
    .groupby("Date", as_index=False)
    .last()
)

print("\nProcessed Repo Dataset")
print(repo.head())
print(repo.tail())
print(repo.shape)



# ==========================================================
# Merge all datasets
# ==========================================================

master = nifty.merge(
    usdinr,
    on="Date",
    how="left"
)

master = master.merge(
    brent,
    on="Date",
    how="left"
)

master = master.merge(
    cpi,
    on="Date",
    how="left"
)

master = master.merge(
    repo,
    on="Date",
    how="left"
)

# ==========================================================
# Forward-fill Repo Rate
# ==========================================================

master["Repo"] = master["Repo"].ffill()

# ==========================================================
# Check for missing values
# ==========================================================

print("\nMissing Values")
print(master.isnull().sum())

# ==========================================================
# Display dataset
# ==========================================================

print("\nMaster Dataset")
print(master.head())
print(master.tail())

print("\nShape")
print(master.shape)

# ==========================================================
# Save
# ==========================================================

master.to_csv(
    PROCESSED_DATA / "master_dataset.csv",
    index=False
)

print("\nMaster dataset saved successfully!")