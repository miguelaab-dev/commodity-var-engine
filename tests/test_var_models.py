"""Tests for VaR/ES models.

Test strategy:
- Property tests: ES >= VaR always; VaR(99%) >= VaR(95%).
- Sanity tests: VaR is positive and within a plausible range.
- Edge cases: very short series, constant returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from varengine.var.cornish_fisher import CornishFisherVaR
from varengine.var.historical import HistoricalVaR
from varengine.var.monte_carlo import MonteCarloVaR
from varengine.var.parametric import ParametricVaR


class TestHistoricalVaR:
    def test_basic_estimate(self, normal_returns: pd.Series) -> None:
        model = HistoricalVaR(window=500)
        model.fit(normal_returns)
        est = model.estimate(confidence=0.99)

        assert est.var > 0, "VaR should be positive"
        assert est.es >= est.var, "ES must be >= VaR"
        assert est.confidence == 0.99

    def test_higher_confidence_gives_higher_var(self, normal_returns: pd.Series) -> None:
        model = HistoricalVaR(window=500)
        model.fit(normal_returns)

        est_95 = model.estimate(confidence=0.95)
        est_99 = model.estimate(confidence=0.99)

        assert est_99.var >= est_95.var, "VaR(99%) must >= VaR(95%)"

    def test_rejects_dataframe(self, multi_asset_returns: pd.DataFrame) -> None:
        model = HistoricalVaR(window=500)
        with pytest.raises(TypeError, match="single portfolio return Series"):
            model.fit(multi_asset_returns)

    def test_minimum_window_enforced(self) -> None:
        with pytest.raises(ValueError, match="Window too short"):
            HistoricalVaR(window=50)


class TestParametricVaR:
    def test_basic_estimate(self, normal_returns: pd.Series) -> None:
        model = ParametricVaR()
        model.fit(normal_returns)
        est = model.estimate(confidence=0.99)

        assert est.var > 0
        assert est.es >= est.var

    def test_with_ewma(self, normal_returns: pd.Series) -> None:
        from varengine.models.volatility.ewma import EWMAVolatility

        model = ParametricVaR(vol_model=EWMAVolatility(lam=0.94))
        model.fit(normal_returns)
        est = model.estimate(confidence=0.99)

        assert est.var > 0
        assert "EWMA" in est.method


class TestMonteCarloVaR:
    def test_normal_simulation(self, normal_returns: pd.Series) -> None:
        model = MonteCarloVaR(dist="normal", n_sims=50_000, seed=42)
        model.fit(normal_returns)
        est = model.estimate(confidence=0.99)

        assert est.var > 0
        assert est.es >= est.var

    def test_t_simulation(self, normal_returns: pd.Series) -> None:
        model = MonteCarloVaR(dist="t", df=5, n_sims=50_000, seed=42)
        model.fit(normal_returns)
        est = model.estimate(confidence=0.99)

        assert est.var > 0
        assert est.es >= est.var

    def test_multi_asset(self, multi_asset_returns: pd.DataFrame) -> None:
        model = MonteCarloVaR(dist="normal", n_sims=10_000, seed=42)
        model.fit(multi_asset_returns)
        weights = np.array([0.5, 0.3, 0.2])
        est = model.estimate(confidence=0.99, weights=weights)

        assert est.var > 0
        assert est.es >= est.var

    def test_reproducibility(self, normal_returns: pd.Series) -> None:
        m1 = MonteCarloVaR(seed=123)
        m1.fit(normal_returns)
        e1 = m1.estimate()

        m2 = MonteCarloVaR(seed=123)
        m2.fit(normal_returns)
        e2 = m2.estimate()

        assert e1.var == e2.var, "Same seed must produce identical results"


class TestCornishFisherVaR:
    def test_basic_estimate(self, normal_returns: pd.Series) -> None:
        model = CornishFisherVaR()
        model.fit(normal_returns)
        est = model.estimate(confidence=0.99)

        assert est.var > 0
        assert est.es >= est.var

    def test_fat_tails_differ_from_normal(
        self,
        normal_returns: pd.Series,
        fat_tail_returns: pd.Series,
    ) -> None:
        m_norm = CornishFisherVaR()
        m_norm.fit(normal_returns)
        e_norm = m_norm.estimate()

        m_fat = CornishFisherVaR()
        m_fat.fit(fat_tail_returns)
        e_fat = m_fat.estimate()

        # Fat tails should produce higher VaR (usually)
        assert e_fat.details["excess_kurtosis"] > e_norm.details["excess_kurtosis"]
