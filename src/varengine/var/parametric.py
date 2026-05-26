"""Parametric (variance-covariance) VaR and Expected Shortfall.

Assumes returns are normally distributed with mean μ and volatility σ
estimated from data.  Optionally uses EWMA conditional volatility
instead of the flat sample standard deviation.

VaR = μ − z_α · σ       (z_α = normal quantile)
ES  = μ + σ · φ(z_α) / (1 − α)    (analytical under normality)

Strengths: fast, closed-form, well-understood.
Weaknesses: normality assumption underestimates tail risk for
commodities (fat tails, skewness).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from varengine.models.volatility.ewma import EWMAVolatility
from varengine.protocols import RiskEstimate, VolatilityModel


class ParametricVaR:
    """Variance-covariance VaR under normality."""

    def __init__(
        self,
        vol_model: VolatilityModel | None = None,
    ) -> None:
        """
        Parameters
        ----------
        vol_model : optional VolatilityModel (e.g. EWMAVolatility).
            If None, uses unconditional sample σ.
        """
        self.vol_model = vol_model
        self._mu: float | None = None
        self._sigma: float | None = None

    def fit(self, returns: pd.Series | pd.DataFrame) -> None:
        if isinstance(returns, pd.DataFrame):
            raise TypeError("ParametricVaR expects a single Series.")

        self._mu = float(returns.mean())

        if self.vol_model is not None:
            self.vol_model.fit(returns)
            self._sigma = float(self.vol_model.forecast(horizon=1)[0])
        else:
            self._sigma = float(returns.std(ddof=1))

    def estimate(self, confidence: float = 0.99) -> RiskEstimate:
        if self._mu is None or self._sigma is None:
            raise RuntimeError("Call fit() before estimate().")

        z = sp_stats.norm.ppf(confidence)
        var = -(self._mu - z * self._sigma)  # positive = loss

        # Analytical ES under normality
        phi_z = sp_stats.norm.pdf(z)
        es = -self._mu + self._sigma * phi_z / (1 - confidence)

        return RiskEstimate(
            var=var,
            es=es,
            confidence=confidence,
            method=self.name,
            details={
                "mu": self._mu,
                "sigma": self._sigma,
                "vol_model": self.vol_model.name if self.vol_model else "sample",
            },
        )

    @property
    def name(self) -> str:
        vol_label = self.vol_model.name if self.vol_model else "SampleVol"
        return f"Parametric-{vol_label}"
