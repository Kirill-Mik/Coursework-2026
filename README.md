# Predicting stock returns from dividend policy

Coursework, HSE University, Faculty of Computer Science, BSc Data Science and
Business Analytics, 2026.
Author: Kirill Mikhelev. Supervisor: Daria Bashminova, Senior Lecturer, HSE FCS.

Question: does a company's dividend policy carry information about its future
stock returns beyond standard financial characteristics?

Answer: not at a yearly horizon. Dividend policy separates firms by risk, but
it does not predict returns.

## Data

| | |
|---|---|
| Source | Yahoo Finance via `yfinance` |
| Universe | S&P constituents (`list of companies.csv`) |
| Period | 10 years, quarterly panel |
| Output | `div_panel_final.csv`, about 8.7 MB |
| Features | 17: dividend yield and payout, volatility, beta, quarter-on-quarter growth, lags, log transforms, interactions (vol x beta, vol x qoq, beta x qoq) |

Collection, cleaning and missing-value handling are in the notebook; the panel
can be rebuilt from the ticker list.

## Method

1. Exploratory analysis and data quality: missing values, duplicates,
   distributions, correlations.
2. Hypothesis tests, each chosen after checking its assumptions:

| | Hypothesis | Test | Result |
|---|---|---|---|
| H1 | Dividend payers and non-payers have equal beta | Shapiro-Wilk rejected normality, so Mann-Whitney U instead of a t-test | Rejected: non-payers have about 19% higher beta (1.21 vs 0.98) |
| H2 | Dividend characteristics explain returns | OLS; Durbin-Watson 0.451 showed autocorrelation, refit with standard errors clustered by ticker | Coefficients lose significance with clustering |
| H3 | Dividend metrics Granger-cause returns | ADF and KPSS for stationarity, then Granger causality (maxlag = 4) | No causal direction found |

3. Models: OLS baseline, Ridge, Lasso, ElasticNet, Random Forest, XGBoost,
   LightGBM, `StackingRegressor`. Tuning with `RandomizedSearchCV`,
   preprocessing in a `Pipeline` with `RobustScaler`.
4. Validation: time-based holdout (split at the 80th percentile of date) as the
   main protocol, `TimeSeriesSplit` and `KFold` as robustness checks. A random
   split would leak future information on a time series.

## Result

R² is between -0.05 and about 0 for every model family. Gradient boosting does
not beat OLS. This agrees with the tests: dividend policy is informative about
risk (H1) but not about future returns (H2, H3). A tuned metric above zero here
would have meant overfitting to the test set.

## Files

```
Coursework_Mikhelev_Kirill_245.ipynb   full study: collection, EDA, hypotheses, models, conclusions
div_panel_final.csv                    final quarterly panel
list of companies.csv                  ticker universe
```

Requirements: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`,
`statsmodels`, `scikit-learn`, `xgboost`, `lightgbm`, `yfinance`.
