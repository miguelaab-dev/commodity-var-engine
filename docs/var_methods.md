# VaR and ES Methodology

## Return Convention

This engine uses **arithmetic returns** (simple percentage changes) rather than logarithmic returns.

**Rationale:** On April 20, 2020, WTI crude oil settled at −$37.63/barrel. Log returns are undefined for zero or negative prices (`ln(P_t / P_{t-1})` produces `NaN` when either price is ≤ 0). Since this engine targets commodity futures, arithmetic returns are the only safe choice.

Arithmetic returns are defined as:

    r_t = (P_t − P_{t−1}) / P_{t−1}

Convention: negative returns represent losses.

## Method 1: Historical Simulation

The simplest non-parametric approach. VaR at confidence level α is the empirical (1−α) quantile of the observed return distribution over a rolling window.

    VaR_α = −Quantile(r, 1−α)

ES is the conditional mean of losses exceeding VaR:

    ES_α = E[−r | −r ≥ VaR_α]

**Configurable parameter:** window size (250d, 500d, 750d). Larger windows give more stable estimates but react slower to regime changes.

## Method 2: Parametric (Variance-Covariance)

Assumes returns are normally distributed with mean μ and volatility σ:

    VaR_α = −(μ − z_α · σ)
    ES_α  = −μ + σ · φ(z_α) / (1 − α)

where z_α is the normal quantile and φ is the standard normal PDF.

Volatility σ can come from:
- **Sample standard deviation** (unconditional, flat)
- **EWMA** (exponentially weighted, λ = 0.94 per RiskMetrics)

**Limitation:** The normality assumption underestimates tail risk for commodities, which exhibit excess kurtosis and skewness.

## Method 3: Monte Carlo Simulation

Simulates N paths of future returns by combining:

1. A **volatility model** (Constant, EWMA, or GARCH) that determines σ_t
2. An **innovation distribution** (Normal or Student-t) that determines the shape of ε_t

    r_t = μ + σ_t · ε_t

For multi-asset portfolios, cross-asset dependence is introduced via Cholesky decomposition of the correlation matrix.

VaR and ES are then computed from the empirical distribution of the simulated portfolio returns.

**Note on GARCH:** GARCH is a conditional variance model, not a distribution. It can be combined with either Normal or Student-t innovations. The Student-t innovation with GARCH is recommended for commodities due to empirically observed volatility clustering and fat tails.

## Method 4: Cornish-Fisher Expansion

A parametric correction to the normal VaR that accounts for skewness (S) and excess kurtosis (K):

    z_CF = z + (z²−1)·S/6 + (z³−3z)·K/24 − (2z³−5z)·S²/36

    VaR_CF = −(μ − z_CF · σ)

**Known limitation:** For extremely non-normal distributions (|S| > 2 or K > 10), the expansion can produce nonsensical results. The implementation includes a sanity check and raises an error rather than returning invalid estimates.

## Expected Shortfall

ES (also called CVaR or Conditional VaR) is the expected loss given that VaR has been breached:

    ES_α = E[Loss | Loss > VaR_α]

ES is the preferred risk metric post-Basel III/FRTB because:
- It is **sub-additive** (diversification always reduces risk), unlike VaR
- It captures the **shape of the tail** beyond the VaR threshold
- It is a **coherent risk measure** in the mathematical sense

## Futures Roll Dates

Continuous futures contracts (`CL=F`, `NG=F`, etc.) show artificial price jumps when the front-month contract rolls to the next expiry. These are not real market moves and inflate VaR estimates.

The engine detects probable roll dates by flagging single-day returns exceeding 6 standard deviations from the mean. These are flagged but not automatically removed — the user decides how to handle them.

## References

- Kupiec, P. (1995). "Techniques for Verifying the Accuracy of Risk Measurement Models." *Journal of Derivatives*.
- Christoffersen, P. (1998). "Evaluating Interval Forecasts." *International Economic Review*.
- Basel Committee (1996, rev. 2006). "Supervisory framework for the use of backtesting."
- Higham, N. (2002). "Computing the nearest correlation matrix." *IMA Journal of Numerical Analysis*.
- J.P. Morgan (1996). *RiskMetrics Technical Document*, 4th ed.
