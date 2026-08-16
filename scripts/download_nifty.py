import yfinance as yf

nifty = yf.download(
    "^NSEI",
    start="2015-01-01",
    end="2026-01-01"
)

print(nifty.head())

nifty.to_csv("nifty_daily.csv")


print(nifty.tail())

