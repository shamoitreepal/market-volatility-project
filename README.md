# Market Volatility Project

## Overview

This project studies the behaviour of the Indian stock market, using the NIFTY 50 index as the main market indicator.

The project started with an analysis of NIFTY returns and their relationship with selected macroeconomic variables such as Brent crude oil prices, the USD/INR exchange rate, inflation, and the repo rate.

The project was later extended to study **volatility dynamics and volatility forecasting** using GARCH-family models.

The main objective of the extension is to understand whether NIFTY volatility is persistent, whether negative shocks affect volatility differently from positive shocks, and how well different models forecast future volatility.

---

## Data

The analysis uses data from 2015 to 2025.

The main variables include:

- NIFTY 50 index
- USD/INR exchange rate
- Brent crude oil prices
- Consumer Price Index (CPI)
- Repo rate

Daily NIFTY data are used for the volatility modelling, while the macroeconomic analysis is conducted at the monthly frequency.

---

## Methodology

The project follows several stages.

### 1. Data preparation

The raw financial and macroeconomic data were cleaned and converted into consistent daily and monthly datasets.

Daily NIFTY returns were calculated from the index prices.

### 2. Testing for volatility clustering

An ARCH-LM test was conducted to check whether the variance of NIFTY returns changes over time.

The test provided strong evidence of ARCH effects, motivating the use of GARCH-type models.

### 3. Volatility modelling

Three models were estimated:

- GARCH(1,1)
- GJR-GARCH(1,1)
- EGARCH(1,1)

GJR-GARCH and EGARCH were included to examine whether negative and positive market shocks have different effects on volatility.

### 4. Model comparison

The models were compared using:

- Log-likelihood
- AIC
- BIC

EGARCH(1,1) provided the lowest AIC and BIC, although its improvement over GJR-GARCH was very small.

### 5. Volatility forecasting

A 30-day volatility forecast was generated using the EGARCH model.

The forecasting performance was also evaluated using an out-of-sample test period.

A simple 20-day rolling variance model was used as a benchmark.

### 6. Macroeconomic determinants of volatility

Monthly realized NIFTY volatility was constructed from daily returns and merged with the monthly macroeconomic dataset.

A regression model was then estimated with lagged volatility and macroeconomic variables as explanatory variables.

HAC (Newey-West) standard errors were used to account for serial correlation and heteroskedasticity.

---

## Main Findings

The analysis gives several important results.

### Volatility is persistent

The GARCH-family models show strong persistence in NIFTY volatility.

The monthly analysis also finds a positive and highly significant coefficient on lagged volatility.

This suggests that periods of high volatility tend to be followed by further periods of relatively high volatility.

### Volatility is asymmetric

Both GJR-GARCH and EGARCH provide evidence that negative shocks have a stronger effect on volatility than positive shocks of a similar magnitude.

This suggests an asymmetric response of NIFTY volatility to market shocks.

### EGARCH provides the best in-sample fit

Among the three volatility models, EGARCH has the lowest AIC and BIC.

However, the difference between EGARCH and GJR-GARCH is very small, so the result should not be interpreted as a large superiority of EGARCH.

### Forecasting results are mixed

EGARCH performs slightly better than the 20-day rolling variance benchmark in terms of MSE and RMSE.

However, the rolling variance benchmark performs better in terms of MAE and QLIKE.

Therefore, the more complicated EGARCH model does not consistently outperform the simple benchmark in out-of-sample forecasting.

### Macroeconomic variables

After controlling for lagged volatility:

- FX returns have a positive and statistically significant relationship with NIFTY volatility.
- Oil returns are not statistically significant.
- Inflation has a negative but relatively weak statistical association with volatility.
- The real repo rate also has a negative and relatively weak statistical association with volatility.

These results should be interpreted as **associations rather than causal effects**.

---

## Project Structure

```text
market_volatility_project/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── scripts/
│
├── outputs/
│   ├── figures/
│   └── results/
│
└── README.md