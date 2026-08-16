import yfinance as yf

brent = yf.download(
    "BZ=F",
    start="2015-01-01",
    end="2026-01-01"
)

print(brent.head())
print(brent.shape)

brent.to_csv("brent_daily.csv")
print(brent.shape)
print(brent.isnull().sum())