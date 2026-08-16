import pandas as pd
import numpy as np
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load data
df = pd.read_csv("data/processed/nifty_daily_returns.csv")

# Get returns
returns = df["Return"].dropna().reset_index(drop=True)

# Split into training and testing data
split = int(len(returns) * 0.80)

train = returns[:split]
test = returns[split:]

print("Training observations:", len(train))
print("Testing observations:", len(test))

# Estimate EGARCH using training data
model = arch_model(
    train,
    mean="Constant",
    vol="EGARCH",
    p=1,
    o=1,
    q=1,
    dist="normal"
)

results = model.fit(disp="off")

print("\nEGARCH model estimated successfully.")

# Forecast volatility for the test period
forecast = results.forecast(
    horizon=len(test),
    method="simulation",
    simulations=1000,
    reindex=False
)

# Get forecasted variance
forecast_variance = forecast.variance.iloc[-1].values

# Actual variance proxy
actual_variance = test.values ** 2

# Calculate forecast errors
mse = mean_squared_error(
    actual_variance,
    forecast_variance
)

rmse = np.sqrt(mse)

mae = mean_absolute_error(
    actual_variance,
    forecast_variance
)

# QLIKE
epsilon = 1e-8

actual_variance_safe = np.maximum(actual_variance, epsilon)
forecast_variance_safe = np.maximum(forecast_variance, epsilon)

qlike = np.mean(
    (actual_variance_safe / forecast_variance_safe)
    - np.log(actual_variance_safe / forecast_variance_safe)
    - 1
)
# Display results
print("\nOUT-OF-SAMPLE FORECAST EVALUATION")
print("=================================")

print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)
print("QLIKE:", qlike)

# Save results
results_df = pd.DataFrame({
    "Actual_Variance": actual_variance,
    "Forecasted_Variance": forecast_variance
})

results_df.to_csv(
    "output/results/egarch_out_of_sample_results.csv",
    index=False
)

print("\nResults saved successfully.")