"""Monte Carlo VaR and Expected Shortfall.

Architecture: Monte Carlo is a *simulation framework*, not a single model.
It composes two independent choices:

1. **Volatility model** — how σ_t evolves (Constant, EWMA, GARCH).
2. **Innovation distribution** — the shape of ε_t (Normal, Student-t).

   r_t = σ_t · ε_t

This separation is the key design insight: you can plug any VolatilityModel
into the simulation without changing the MC engine itself.

For multi-asset portfolios: we simulate each asset independently,
apply the Cholesky decomposition of the correlation matrix to introduce
cross-asset dependence, then weight by portfolio weights.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy.linalg import cholesky

from varengine.models.volatility.constant import ConstantVolatility
from varengine.protocols import RiskEstimate, VolatilityModel


class MonteCarloVaR:
    """Monte Carlo VaR with configurable volatility and innovation distribution."""

    def __init__(
        self,
        vol_model: VolatilityModel | None = None,
        dist: Literal["normal", "t"] = "normal",
        df: int = 5,
        n_sims: int = 10_000,
        seed: int | None = 42,
    ) -> None:
        """
        Parameters
        ----------
        vol_model : VolatilityModel to forecast σ.  Defaults to ConstantVolatility.
        dist : Innovation distribution ('normal' or 't').
        df : Degrees of freedom for Student-t (ignored if dist='normal').
        n_sims : Number of Monte Carlo paths.
        seed : Random seed for reproducibility.
        """
        self.vol_model = vol_model or ConstantVolatility()
        self.dist = dist
        self.df = df
        self.n_sims = n_sims
        self.seed = seed
        self._simulated_returns: np.ndarray | None = None

    def fit(self, returns: pd.Series | pd.DataFrame) -> None:
        """Fit the volatility model and run simulation."""
        if isinstance(returns, pd.DataFrame):
            # Multi-asset: fit each, simulate with correlation
            self._fit_portfolio(returns)
        else:
            self._fit_single(returns)

    def _fit_single(self, returns: pd.Series) -> None:
        self.vol_model.fit(returns)
        sigma = self.vol_model.forecast(horizon=1)[0]

        rng = np.random.default_rng(self.seed)

        if self.dist == "normal":
            innovations = rng.standard_normal(self.n_sims)
        elif self.dist == "t":
            # Scale t-distribution so variance = 1
            raw = rng.standard_t(self.df, size=self.n_sims)
            innovations = raw / np.sqrt(self.df / (self.df - 2))
        else:
            raise ValueError(f"Unknown distribution: {self.dist}")

        mu = float(returns.mean())
        self._simulated_returns = mu + sigma * innovations

    def _fit_portfolio(self, returns: pd.DataFrame) -> None:
        """Simulate correlated multi-asset returns via Cholesky decomposition."""
        n_assets = returns.shape[1]
        corr = returns.corr().values

        # Ensure PSD (Higham nearest-PSD correction if needed)
        try:
            L = cholesky(corr, lower=True)
        except np.linalg.LinAlgError:
            corr = self._nearest_psd(corr)
            L = cholesky(corr, lower=True)

        rng = np.random.default_rng(self.seed)

        if self.dist == "normal":
            Z = rng.standard_normal((self.n_sims, n_assets))
        else:
            raw = rng.standard_t(self.df, size=(self.n_sims, n_assets))
            Z = raw / np.sqrt(self.df / (self.df - 2))

        # Correlate the innovations
        correlated = Z @ L.T

        # Scale by per-asset volatility
        sigmas = np.array([returns.iloc[:, i].std(ddof=1) for i in range(n_assets)])
        mus = returns.mean().values

        self._simulated_returns = mus + sigmas * correlated
        # Store for later — portfolio weighting happens in estimate()
        self._is_portfolio = True
        self._columns = list(returns.columns)

    def estimate(
        self,
        confidence: float = 0.99,
        weights: np.ndarray | None = None,
    ) -> RiskEstimate:
        if self._simulated_returns is None:
            raise RuntimeError("Call fit() before estimate().")

        sims = self._simulated_returns

        # If multi-asset, compute weighted portfolio returns
        if hasattr(self, "_is_portfolio") and self._is_portfolio:
            if weights is None:
                # Equal weight fallback
                weights = np.ones(sims.shape[1]) / sims.shape[1]
            sims = sims @ weights

        var = -float(np.quantile(sims, 1 - confidence))
        losses = -sims
        es = float(losses[losses >= var].mean())

        return RiskEstimate(
            var=var,
            es=es,
            confidence=confidence,
            method=self.name,
            details={
                "n_sims": self.n_sims,
                "vol_model": self.vol_model.name,
                "dist": self.dist,
                "df": self.df if self.dist == "t" else None,
            },
        )

    @staticmethod
    def _nearest_psd(matrix: np.ndarray) -> np.ndarray:
        """Nearest positive semi-definite matrix (Higham 2002 simplified)."""
        eigvals, eigvecs = np.linalg.eigh(matrix)
        eigvals = np.maximum(eigvals, 1e-8)
        psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        # Re-normalise to correlation matrix
        d = np.sqrt(np.diag(psd))
        psd = psd / np.outer(d, d)
        return psd

    @property
    def name(self) -> str:
        dist_label = f"t(df={self.df})" if self.dist == "t" else "Normal"
        return f"MonteCarlo-{self.vol_model.name}-{dist_label}"
