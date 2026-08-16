import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load daily returns
df = pd.read_csv("data/processed/nifty_daily_returns.csv")

# Get returns
returns = df["Return"].dropna().reset_index(drop=True)

# Split data
split = int(len(returns) * 0.80)

train = returns[:split]
test = returns[split:]

print("Training observations:", len(train))
print("Testing observations:", len(test))

# Combine training and testing returns
all_returns = returns.values

# Store benchmark forecasts
benchmark_variance = []

# 20-day rolling variance forecast
window = 20

for i in range(split, len(all_returns)):

    # Previous 20 observations
    previous_returns = all_returns[i - window:i]

    # Calculate average squared return
    variance = np.mean(previous_returns ** 2)

    benchmark_variance.append(variance)

benchmark_variance = np.array(benchmark_variance)

# Actual variance proxy
actual_variance = test.values ** 2

# Calculate MSE
mse = mean_squared_error(
    actual_variance,
    benchmark_variance
)

# Calculate RMSE
rmse = np.sqrt(mse)

# Calculate MAE
mae = mean_absolute_error(
    actual_variance,
    benchmark_variance
)

# Calculate QLIKE
epsilon = 1e-8

actual_safe = np.maximum(actual_variance, epsilon)
benchmark_safe = np.maximum(benchmark_variance, epsilon)

qlike = np.mean(
    (actual_safe / benchmark_safe)
    - np.log(actual_safe / benchmark_safe)
    - 1
)

# Display results
print("\n20-DAY ROLLING VARIANCE BENCHMARK")
print("=================================")

print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)
print("QLIKE:", qlike)

# Save results
benchmark_results = pd.DataFrame({
    "Model": ["20-Day Rolling Variance"],
    "MSE": [mse],
    "RMSE": [rmse],
    "MAE": [mae],
    "QLIKE": [qlike]
})

benchmark_results.to_csv(
    "output/results/benchmark_results.csv",
    index=False
)

print("\nBenchmark results saved successfully.")