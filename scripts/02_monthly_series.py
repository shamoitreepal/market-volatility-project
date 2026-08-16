from pathlib import Path
import pandas as pd

# ==========================================================
# Define project folders
# ==========================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

PROCESSED_DATA = PROJECT_DIR / "data" / "processed"

# ==========================================================
# Function to convert daily data to monthly
# ==========================================================

def create_monthly_series(input_file, output_file):

    print(f"\nProcessing: {input_file.name}")

    # Read cleaned data
    df = pd.read_csv(input_file)

    # Convert Date column to datetime
    df["Date"] = pd.to_datetime(df["Date"])

    # Set Date as index
    df.set_index("Date", inplace=True)

    # Take the last trading day of every month
    monthly = df.resample("ME").last()

    # Bring Date back as a column
    monthly.reset_index(inplace=True)

    # Save
    monthly.to_csv(output_file, index=False)

    print(monthly.head())
    print(monthly.tail())
    print(f"Saved to: {output_file}")

# ==========================================================
# Convert all datasets
# ==========================================================

create_monthly_series(
    PROCESSED_DATA / "nifty_clean.csv",
    PROCESSED_DATA / "nifty_monthly.csv"
)

create_monthly_series(
    PROCESSED_DATA / "usdinr_clean.csv",
    PROCESSED_DATA / "usdinr_monthly.csv"
)

create_monthly_series(
    PROCESSED_DATA / "brent_clean.csv",
    PROCESSED_DATA / "brent_monthly.csv"
)

print("\nMonthly datasets created successfully!")