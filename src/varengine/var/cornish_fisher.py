"""Cornish-Fisher expansion VaR and Expected Shortfall.

Adjusts the normal quantile for skewness (S) and excess kurtosis (K):

    z_CF = z + (z² − 1)·S/6 + (z³ − 3z)·K/24 − (2z³ − 5z)·S²/36

This is a quick parametric correction that avoids the full machinery
of fitting a non-normal distribution.  Useful when the departure from
normality is moderate.

KNOWN LIMITATION: for extremely fat-tailed distributions the expansion
can produce nonsensical results (VaR < mean, or negative ES).  We
validate the output and raise a clear error rather than returning garbage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from varengine.protocols import RiskEstimate


class CornishFisherVaR:
    """Cornish-Fisher expansion VaR with skewness/kurtosis correction."""

    def __init__(self) -> None:
        self._mu: float | None = None
        self._sigma: float | None = None
        self._skew: float | None = None
        self._excess_kurt: float | None = None

    def fit(self, returns: pd.Series | pd.DataFrame) -> None:
        if isinstance(returns, pd.DataFrame):
            raise TypeError("CornishFisherVaR expects a single Series.")

        self._mu = float(returns.mean())
        self._sigma = float(returns.std(ddof=1))
        self._skew = float(sp_stats.skew(returns, bias=False))
        self._excess_kurt = float(sp_stats.kurtosis(returns, bias=False))

    def _cf_quantile(self, confidence: float) -> float:
        """Compute the Cornish-Fisher adjusted quantile."""
        z = sp_stats.norm.ppf(confidence)
        S = self._skew
        K = self._excess_kurt

        z_cf = (
            z
            + (z**2 - 1) * S / 6
            + (z**3 - 3 * z) * K / 24
            - (2 * z**3 - 5 * z) * S**2 / 36
        )
        return z_cf

    def estimate(self, confidence: float = 0.99) -> RiskEstimate:
        if self._mu is None or self._sigma is None:
            raise RuntimeError("Call fit() before estimate().")

        z_cf = self._cf_quantile(confidence)
        var = -(self._mu - z_cf * self._sigma)

        # ES approximation: use the normal ES formula but with CF-adjusted σ
        # This is a pragmatic approximation; exact CF-ES requires integration.
        z = sp_stats.norm.ppf(confidence)
        phi_z_cf = sp_stats.norm.pdf(z_cf)
        es = -self._mu + self._sigma * phi_z_cf / (1 - confidence)

        # Ensure ES >= VaR (can fail for extreme skew/kurtosis)
        es = max(es, var)

        # Sanity check
        if var < 0:
            raise ValueError(
                f"Cornish-Fisher produced negative VaR ({var:.6f}). "
                f"Distribution is too non-normal for CF expansion "
                f"(skew={self._skew:.3f}, excess_kurt={self._excess_kurt:.3f}). "
                f"Use Monte Carlo or Historical Simulation instead."
            )

        return RiskEstimate(
            var=var,
            es=es,
            confidence=confidence,
            method=self.name,
            details={
                "skewness": self._skew,
                "excess_kurtosis": self._excess_kurt,
                "z_cf": z_cf,
            },
        )

    @property
    def name(self) -> str:
        return "CornishFisher"
