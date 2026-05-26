"""Constant (unconditional) volatility — the simplest baseline.

σ_t = sample standard deviation of the estimation window.
No time-variation, no clustering.  Useful as a benchmark to show
that EWMA/GARCH actually improve forecasts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class ConstantVolatility:
    """Unconditional sample standard deviation."""

    def __init__(self) -> None:
        self._sigma: float | None = None

    def fit(self, returns: pd.Series) -> None:
        self._sigma = float(returns.std(ddof=1))

    def forecast(self, horizon: int = 1) -> np.ndarray:
        if self._sigma is None:
            raise RuntimeError("Call fit() before forecast().")
        return np.full(horizon, self._sigma)

    @property
    def name(self) -> str:
        return "Constant"
