import pandas as pd
from arch import arch_model

# Load daily returns
df = pd.read_csv("data/processed/nifty_daily_returns.csv")

# Remove missing values
returns = df["Return"].dropna()

# Estimate GJR-GARCH(1,1)
model = arch_model(
    returns,
    mean="Constant",
    vol="GARCH",
    p=1,
    o=1,
    q=1,
    dist="normal"
)

# Fit the model
results = model.fit(disp="off")

# Display results
print(results.summary())