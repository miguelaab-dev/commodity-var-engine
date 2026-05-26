# Backtesting Methodology

## Walk-Forward Framework

All backtesting in this engine is **out-of-sample**. For each day t:

1. Fit the VaR model on the estimation window `[t−w, t]`
2. Produce an ex-ante VaR forecast for day `t+1`
3. Observe the realised return at `t+1`
4. Record whether the loss exceeded the VaR forecast (a "violation")

This produces a time series of violations that can be tested statistically.

**In-sample backtesting** (fitting on the entire dataset and testing against the same data) is not supported because it produces meaningless results — any model can "fit" its own training data.

## Statistical Power Warning

At 99% confidence with 250 observations, the expected number of violations is ~2.5. Statistical tests have essentially no power to distinguish a good model from a bad one with so few events. This engine warns when fewer than 500 backtest observations are available.

## Test 1: Kupiec Proportion of Failures (POF)

Tests H0: the observed violation rate equals the expected rate (1−α).

    LR_POF = −2 ln[ (1−p)^(n−x) · p^x / ((1−p̂)^(n−x) · p̂^x) ]

Under H0, LR_POF ~ χ²(1).

**Interpretation:** rejection means the model produces too many or too few violations — it's either too aggressive or too conservative.

## Test 2: Christoffersen Independence

Tests H0: violations are serially independent (no clustering).

Fits a first-order Markov chain to the violation sequence and tests whether transition probabilities p(violation | no prior violation) and p(violation | prior violation) are equal.

**Interpretation:** rejection means violations cluster in time — the model fails to capture volatility persistence.

## Test 3: Conditional Coverage (CC)

Joint test combining Kupiec + Independence (df=2). Rejection means the model fails on rate, independence, or both.

## Test 4: Basel Traffic Light

A regulatory classification (not a statistical test):

| Zone   | Violations (250d, 99%) | Implication |
|--------|----------------------|-------------|
| Green  | 0–4                  | Model is acceptable |
| Yellow | 5–9                  | Increased capital multiplier |
| Red    | 10+                  | Model is rejected |

## Interpreting Results

A good model should:
- **Pass Kupiec** (violation rate close to expected)
- **Pass Independence** (violations are not clustered)
- **Be in the Green zone** (Basel traffic light)
- Have a **violation ratio** near 1.0 (observed/expected rate)

No model is perfect. The goal is to identify which methods are least wrong for the specific portfolio and time period.
