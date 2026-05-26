"""GARCH(1,1) conditional volatility via the `arch` library.

σ²_t = ω + α · r²_{t-1} + β · σ²_{t-1}

Innovation distribution can be Normal or Student-t.  The t distribution
is preferred for commodities because of empirically observed fat tails
(excess kurtosis).

Wraps Kevin Sheppard's `arch` package, which is the academic standard
for GARCH estimation in Python.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


class GARCHVolatility:
    """GARCH(1,1) estimator backed by the ``arch`` library."""

    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        dist: Literal["normal", "t"] = "t",
    ) -> None:
        self.p = p
        self.q = q
        self.dist = dist
        self._result: object | None = None  # arch ModelResult

    def fit(self, returns: pd.Series) -> None:
        from arch import arch_model  # lazy import — heavy dependency

        # arch expects returns scaled to percentage points
        scaled = returns * 100

        model = arch_model(
            scaled,
            vol="Garch",
            p=self.p,
            q=self.q,
            dist=self.dist if self.dist == "normal" else "StudentsT",
            mean="Zero",  # assume zero-mean for daily returns
        )
        self._result = model.fit(disp="off", show_warning=False)

    def forecast(self, horizon: int = 1) -> np.ndarray:
        """Return forecasted volatility (daily σ, not percentage)."""
        if self._result is None:
            raise RuntimeError("Call fit() before forecast().")

        fc = self._result.forecast(horizon=horizon)
        # arch returns variance in (percentage)^2; convert back
        var_pct = fc.variance.iloc[-1].values[:horizon]
        return np.sqrt(var_pct) / 100

    @property
    def params(self) -> dict[str, float]:
        if self._result is None:
            raise RuntimeError("Call fit() before accessing params.")
        return dict(self._result.params)

    @property
    def conditional_vol(self) -> np.ndarray:
        """In-sample conditional volatility (daily σ)."""
        if self._result is None:
            raise RuntimeError("Call fit() before accessing conditional_vol.")
        return self._result.conditional_volatility.values / 100

    @property
    def name(self) -> str:
        return f"GARCH({self.p},{self.q})-{self.dist}"
