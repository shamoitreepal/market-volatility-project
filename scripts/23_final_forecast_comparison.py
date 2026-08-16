import pandas as pd

# EGARCH results
egarch = {
    "Model": "EGARCH(1,1)",
    "MSE": 4.2293251013,
    "RMSE": 2.0565323001,
    "MAE": 0.9849068749,
    "QLIKE": 1.8535865691
}

# Benchmark results
benchmark = {
    "Model": "20-Day Rolling Variance",
    "MSE": 4.3231799707,
    "RMSE": 2.0792258104,
    "MAE": 0.7961868724,
    "QLIKE": 1.7922091247
}

# Create comparison table
comparison = pd.DataFrame([
    egarch,
    benchmark
])

# Display results
print("\nFINAL FORECAST COMPARISON")
print("=========================")
print(comparison.to_string(index=False))

# Save results
comparison.to_csv(
    "output/results/final_forecast_comparison.csv",
    index=False
)

print("\nFinal comparison saved successfully.")