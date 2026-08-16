import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.stats.diagnostic import het_arch

# Load daily returns
df = pd.read_csv("data/processed/nifty_daily_returns.csv")

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

results = model.fit(disp="off")

# Get standardized residuals
standardized_residuals = results.std_resid.dropna()

# Squared standardized residuals
squared_residuals = standardized_residuals ** 2

# ARCH-LM test on standardized residuals
test = het_arch(
    squared_residuals,
    nlags=10
)

lm_stat = test[0]
lm_pvalue = test[1]

# Display results
print("\nEGARCH RESIDUAL DIAGNOSTIC")
print("-----------------------------")

print("ARCH-LM statistic:", lm_stat)
print("ARCH-LM p-value:", lm_pvalue)

if lm_pvalue < 0.05:
    print("\nARCH effects remain in the residuals.")
    print("The EGARCH model may not fully capture volatility dynamics.")
else:
    print("\nNo significant ARCH effects remain.")
    print("The EGARCH model appears to capture the volatility dynamics adequately.")

# Plot squared standardized residuals
plt.figure(figsize=(12, 5))

plt.plot(
    squared_residuals
)

plt.title("Squared Standardized Residuals - EGARCH")
plt.xlabel("Observation")
plt.ylabel("Squared Standardized Residual")

plt.tight_layout()
plt.show()

# Save diagnostic result
with open(
    "output/results/egarch_diagnostics.txt",
    "w"
) as f:

    f.write("EGARCH RESIDUAL DIAGNOSTIC\n")
    f.write("---------------------------\n\n")
    f.write(f"ARCH-LM statistic: {lm_stat}\n")
    f.write(f"ARCH-LM p-value: {lm_pvalue}\n\n")

    if lm_pvalue < 0.05:
        f.write(
            "Conclusion: ARCH effects remain in the residuals.\n"
        )
    else:
        f.write(
            "Conclusion: No significant ARCH effects remain.\n"
        )

print("\nDiagnostic results successfully saved.")