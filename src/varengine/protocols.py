"""Core protocols and data structures shared across the engine.

These protocols define the contracts that all VaR models, volatility models,
and backtest methods must satisfy. Design new modules against these interfaces
— never import concrete implementations where a protocol would do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RiskEstimate:
    """Single-point VaR and ES estimate produced by a model."""

    var: float
    es: float
    confidence: float
    method: str
    details: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 < self.confidence < 1:
            raise ValueError(f"confidence must be in (0, 1), got {self.confidence}")
        if self.es < self.var:
            raise ValueError(
                f"ES ({self.es:.6f}) must be >= VaR ({self.var:.6f}). "
                "Check model implementation."
            )


@dataclass(frozen=True)
class RollingSeries:
    """Time series of VaR/ES forecasts aligned with realised returns."""

    dates: pd.DatetimeIndex
    var_series: pd.Series       # ex-ante VaR forecasts  (positive = loss)
    es_series: pd.Series        # ex-ante ES forecasts   (positive = loss)
    realised: pd.Series         # ex-post realised returns (negative = loss)
    confidence: float
    method: str


@dataclass(frozen=True)
class BacktestResult:
    """Output of a single statistical backtest."""

    test_name: str
    statistic: float
    p_value: float
    reject_null: bool           # True → model is rejected at the given alpha
    alpha: float                # significance level of the test (e.g. 0.05)
    details: dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Volatility model protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class VolatilityModel(Protocol):
    """Estimates conditional volatility from a return series."""

    def fit(self, returns: pd.Series) -> None:
        """Fit the model to historical returns."""
        ...

    def forecast(self, horizon: int = 1) -> np.ndarray:
        """Return *horizon* steps of forecasted volatility (annualised σ)."""
        ...

    @property
    def name(self) -> str:
        """Short human-readable label, e.g. 'GARCH(1,1)-t'."""
        ...


# ---------------------------------------------------------------------------
# VaR / ES model protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class VaRModel(Protocol):
    """Computes Value-at-Risk and Expected Shortfall for a return series."""

    def fit(self, returns: pd.Series | pd.DataFrame) -> None:
        """Calibrate the model on historical returns.

        Parameters
        ----------
        returns : Series (single asset) or DataFrame (portfolio constituents).
                  Convention: **arithmetic** returns where negative = loss.
        """
        ...

    def estimate(self, confidence: float = 0.99) -> RiskEstimate:
        """Return a point-in-time VaR + ES estimate after fitting.

        Returns
        -------
        RiskEstimate with *positive* VaR/ES representing loss thresholds.
        """
        ...

    @property
    def name(self) -> str:
        """Short label, e.g. 'HistoricalSimulation-500d'."""
        ...


# ---------------------------------------------------------------------------
# Backtest method protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class BacktestMethod(Protocol):
    """Statistical test applied to a RollingSeries."""

    def test(self, series: RollingSeries, alpha: float = 0.05) -> BacktestResult:
        """Run the test and return a result."""
        ...

    @property
    def name(self) -> str:
        ...
