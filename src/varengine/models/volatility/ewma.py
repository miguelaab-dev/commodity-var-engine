"""Exponentially Weighted Moving Average volatility (RiskMetrics).

σ²_t = λ · σ²_{t-1} + (1 − λ) · r²_{t-1}

The classic RiskMetrics decay factor is λ = 0.94 for daily data.
EWMA reacts faster to recent shocks than a flat window while remaining
parameter-free beyond λ.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class EWMAVolatility:
    """EWMA conditional variance estimator."""

    def __init__(self, lam: float = 0.94) -> None:
        if not 0 < lam < 1:
            raise ValueError(f"λ must be in (0, 1), got {lam}")
        self.lam = lam
        self._var_series: np.ndarray | None = None

    def fit(self, returns: pd.Series) -> None:
        r = returns.values
        n = len(r)
        var = np.empty(n)
        var[0] = r[0] ** 2  # seed with first squared return
        for t in range(1, n):
            var[t] = self.lam * var[t - 1] + (1 - self.lam) * r[t - 1] ** 2
        self._var_series = var

    def forecast(self, horizon: int = 1) -> np.ndarray:
        """Forecast next *horizon* periods of volatility (σ, not σ²)."""
        if self._var_series is None:
            raise RuntimeError("Call fit() before forecast().")
        last_var = self._var_series[-1]
        # EWMA forecast is flat (no mean-reversion term)
        return np.full(horizon, np.sqrt(last_var))

    @property
    def conditional_vol(self) -> np.ndarray:
        """Full in-sample conditional volatility series."""
        if self._var_series is None:
            raise RuntimeError("Call fit() before accessing conditional_vol.")
        return np.sqrt(self._var_series)

    @property
    def name(self) -> str:
        return f"EWMA(λ={self.lam})"
