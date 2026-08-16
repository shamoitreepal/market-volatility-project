import pandas as pd
from statsmodels.stats.diagnostic import het_arch

# Load daily returns
df = pd.read_csv("data/processed/nifty_daily_returns.csv")

# Remove missing returns
returns = df["Return"].dropna()

# ARCH-LM test
test = het_arch(returns, nlags=10)

# Extract results
lm_stat = test[0]
lm_pvalue = test[1]
f_stat = test[2]
f_pvalue = test[3]

# Display results
print("ARCH-LM Test")
print("----------------------------")
print("LM Statistic:", lm_stat)
print("LM p-value:", lm_pvalue)
print("F Statistic:", f_stat)
print("F p-value:", f_pvalue)

# Interpretation
print("\nInterpretation:")

if lm_pvalue < 0.05:
    print("Reject H0: ARCH effects are present.")
    print("A GARCH-type model is appropriate.")
else:
    print("Fail to reject H0: No significant ARCH effects are detected.")