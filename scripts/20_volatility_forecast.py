import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model

# Load daily returns
df = pd.read_csv("data/processed/nifty_daily_returns.csv")

# Convert Date to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Get returns
returns = df["Return"].dropna()

# Estimate EGARCH(1,1)
model = arch_model(
    returns,
    mean="Constant",
    vol="EGARCH",
    p=1,
    o=1,
    q=1,
    dist="normal"
)

# Fit the model
results = model.fit(disp="off")

# Forecast volatility for the next 30 days
# EGARCH uses simulation for multi-step forecasts
forecast = results.forecast(
    horizon=30,
    method="simulation",
    simulations=1000
)

# Extract forecasted variance
variance_forecast = forecast.variance.iloc[-1]

# Convert variance to volatility
volatility_forecast = variance_forecast ** 0.5

# Display forecast
print("\n30-DAY VOLATILITY FORECAST")
print("==========================")

for day, volatility in enumerate(volatility_forecast, start=1):
    print(f"Day {day}: {volatility:.4f}%")

# Plot forecast
plt.figure(figsize=(12, 5))

plt.plot(
    range(1, 31),
    volatility_forecast.values
)

plt.title("NIFTY 30-Day Ahead Volatility Forecast")
plt.xlabel("Forecast Horizon (Days)")
plt.ylabel("Forecasted Volatility (%)")

plt.tight_layout()
plt.show()

# Save forecast
forecast_df = pd.DataFrame({
    "Forecast_Day": range(1, 31),
    "Forecasted_Volatility": volatility_forecast.values
})

forecast_df.to_csv(
    "output/results/egarch_30day_forecast.csv",
    index=False
)

print("\nForecast saved successfully.")