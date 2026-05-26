"""Walk-forward (out-of-sample) backtest runner.

This is the most critical module in the backtest package.  It enforces
the correct methodology:

    For each day t in [window, T]:
        1. Fit the model on returns[t-window : t]     (estimation window)
        2. Forecast VaR and ES for day t+1             (ex-ante prediction)
        3. Observe realised return at t+1              (ex-post outcome)
        4. Record whether VaR was breached             (violation)

This produces a RollingSeries that can be fed to any BacktestMethod
(Kupiec, Christoffersen, traffic light).

WARNING: in-sample backtesting (fitting on the full dataset and then
"testing" against the same data) is meaningless.  This module exists
to prevent that mistake.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from varengine.protocols import RiskEstimate, RollingSeries, VaRModel


def walk_forward(
    model: VaRModel,
    returns: pd.Series,
    window: int = 500,
    confidence: float = 0.99,
    min_obs_warning: int = 500,
) -> RollingSeries:
    """Run a walk-forward backtest and return a RollingSeries.

    Parameters
    ----------
    model : any object satisfying the VaRModel protocol.
    returns : portfolio or single-asset return series.
    window : rolling estimation window in days.
    confidence : VaR confidence level.
    min_obs_warning : warn if the backtest produces fewer observations.

    Returns
    -------
    RollingSeries with aligned ex-ante forecasts and ex-post returns.
    """
    n = len(returns)
    if n <= window:
        raise ValueError(
            f"Return series has {n} observations but window is {window}. "
            f"Need at least window + 1 = {window + 1} observations."
        )

    forecast_dates = []
    var_values = []
    es_values = []
    realised_values = []

    for t in range(window, n - 1):
        estimation_window = returns.iloc[t - window : t]
        model.fit(estimation_window)
        est: RiskEstimate = model.estimate(confidence=confidence)

        forecast_dates.append(returns.index[t + 1])
        var_values.append(est.var)
        es_values.append(est.es)
        realised_values.append(returns.iloc[t + 1])

    n_forecasts = len(forecast_dates)
    if n_forecasts < min_obs_warning:
        warnings.warn(
            f"Backtest produced only {n_forecasts} observations. "
            f"Statistical tests (Kupiec, Christoffersen) have low power "
            f"below ~500 observations. Consider a longer return history.",
            stacklevel=2,
        )

    idx = pd.DatetimeIndex(forecast_dates)

    return RollingSeries(
        dates=idx,
        var_series=pd.Series(var_values, index=idx, name="VaR"),
        es_series=pd.Series(es_values, index=idx, name="ES"),
        realised=pd.Series(realised_values, index=idx, name="Realised"),
        confidence=confidence,
        method=model.name,
    )


def count_violations(series: RollingSeries) -> int:
    """Count the number of days where the loss exceeded VaR."""
    losses = -series.realised  # positive = loss
    return int((losses > series.var_series).sum())


def violation_ratio(series: RollingSeries) -> float:
    """Observed violation rate vs expected rate."""
    n_violations = count_violations(series)
    n_obs = len(series.dates)
    expected_rate = 1 - series.confidence
    observed_rate = n_violations / n_obs
    return observed_rate / expected_rate  # ratio near 1.0 is good
