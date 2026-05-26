"""Historical Simulation VaR and Expected Shortfall.

The simplest non-parametric approach: VaR is the empirical quantile
of the observed return distribution; ES is the mean of returns
beyond that quantile.

Strengths: no distributional assumptions, captures fat tails and
skewness implicitly.

Weaknesses: completely driven by the estimation window — if the
window misses a crisis, so does the VaR.  Also, every observation
has equal weight regardless of recency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from varengine.protocols import RiskEstimate


class HistoricalVaR:
    """Historical simulation with configurable rolling window."""

    def __init__(self, window: int = 500) -> None:
        if window < 100:
            raise ValueError(f"Window too short ({window}). Minimum 100 for meaningful HS.")
        self.window = window
        self._returns: pd.Series | None = None

    def fit(self, returns: pd.Series | pd.DataFrame) -> None:
        """Store the most recent *window* observations."""
        if isinstance(returns, pd.DataFrame):
            raise TypeError(
                "HistoricalVaR expects a single portfolio return Series. "
                "Compute portfolio returns first via Portfolio.portfolio_returns()."
            )
        self._returns = returns.iloc[-self.window :]

    def estimate(self, confidence: float = 0.99) -> RiskEstimate:
        if self._returns is None:
            raise RuntimeError("Call fit() before estimate().")

        r = self._returns.values
        # VaR: the (1-confidence) quantile of the loss distribution
        # Convention: returns are signed (negative = loss),
        # VaR is reported as a positive number representing loss threshold.
        var = -float(np.quantile(r, 1 - confidence))

        # ES: mean of losses beyond VaR
        losses = -r
        es = float(losses[losses >= var].mean())

        return RiskEstimate(
            var=var,
            es=es,
            confidence=confidence,
            method=self.name,
            details={"window": self.window, "n_obs": len(r)},
        )

    @property
    def name(self) -> str:
        return f"HistoricalSimulation-{self.window}d"
