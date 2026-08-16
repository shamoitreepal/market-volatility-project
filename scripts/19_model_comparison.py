import pandas as pd

# Model results
results = {
    "Model": [
        "GARCH(1,1)",
        "GJR-GARCH(1,1)",
        "EGARCH(1,1)"
    ],
    "Log-Likelihood": [
        -3458.57,
        -3427.65,
        -3427.51
    ],
    "AIC": [
        6925.13,
        6865.31,
        6865.03
    ],
    "BIC": [
        6948.75,
        6894.83,
        6894.55
    ]
}

# Create comparison table
comparison = pd.DataFrame(results)

# Display table
print("\nMODEL COMPARISON")
print("================")
print(comparison.to_string(index=False))

# Identify best models
best_aic = comparison.loc[comparison["AIC"].idxmin(), "Model"]
best_bic = comparison.loc[comparison["BIC"].idxmin(), "Model"]

print("\nBest model according to AIC:", best_aic)
print("Best model according to BIC:", best_bic)

# Save results
comparison.to_csv(
    "output/results/model_comparison.csv",
    index=False
)

print("\nModel comparison saved successfully.")