import yfinance as yf

usdinr = yf.download(
    "INR=X",
    start="2015-01-01",
    end="2026-01-01"
)

print(usdinr.head())
print(usdinr.shape)

usdinr.to_csv("usdinr_daily.csv")
import pandas as pd

usdinr = pd.read_csv("usdinr_daily.csv")

print(usdinr.isnull().sum())


print(usdinr.head(3))