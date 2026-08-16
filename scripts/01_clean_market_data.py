from pathlib import Path
import pandas as pd

# ==========================================================
# Define project folders
# ==========================================================

# This file is inside: market_volatility_project/scripts/
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

RAW_DATA = PROJECT_DIR / "data" / "raw"
PROCESSED_DATA = PROJECT_DIR / "data" / "processed"

# Create processed folder if it doesn't exist
PROCESSED_DATA.mkdir(exist_ok=True)

# ==========================================================
# Function to clean Yahoo Finance data
# ==========================================================

def clean_market_data(input_file, output_file, dayfirst):

    print(f"\nReading: {input_file}")

    # Read CSV
    df = pd.read_csv(input_file)

    # Remove first two metadata rows
    df = df.iloc[2:].reset_index(drop=True)

    # Rename columns
    df.columns = [
        "Date",
        "Close",
        "High",
        "Low",
        "Open",
        "Volume"
    ]
    print(df["Date"].head(10))
    # Convert Date
    df["Date"] = pd.to_datetime(
    df["Date"],
    dayfirst=dayfirst
)

    # Convert numeric columns
    numeric_cols = ["Close", "High", "Low", "Open", "Volume"]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])

    # Keep only Date and Close
    df = df[["Date", "Close"]]

    # Sort by date
    df = df.sort_values("Date")

    # Save
    df.to_csv(output_file, index=False)

    print(f"Saved: {output_file}")
    print(df.head())
    print(df.tail())


# ==========================================================
# Run for all datasets
# ==========================================================

clean_market_data(
    RAW_DATA / "nifty_daily.csv",
    PROCESSED_DATA / "nifty_clean.csv",
    dayfirst=True
)

clean_market_data(
    RAW_DATA / "usdinr_daily.csv",
    PROCESSED_DATA / "usdinr_clean.csv",
    dayfirst=False
)

clean_market_data(
    RAW_DATA / "brent_daily.csv",
    PROCESSED_DATA / "brent_clean.csv",
    dayfirst=False
)

print("\nAll market datasets cleaned successfully!")