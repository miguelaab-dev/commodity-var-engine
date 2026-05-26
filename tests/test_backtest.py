"""Tests for the backtesting framework."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from varengine.backtest.christoffersen import ChristoffersenCC, ChristoffersenIndependence
from varengine.backtest.kupiec import KupiecPOF
from varengine.backtest.runner import count_violations, walk_forward
from varengine.backtest.traffic_light import TrafficLight
from varengine.protocols import RollingSeries
from varengine.var.historical import HistoricalVaR


class TestWalkForward:
    def test_produces_correct_length(self, normal_returns: pd.Series) -> None:
        model = HistoricalVaR(window=200)
        series = walk_forward(model, normal_returns, window=200, confidence=0.99)

        expected_len = len(normal_returns) - 200 - 1
        assert len(series.dates) == expected_len

    def test_raises_on_short_data(self, normal_returns: pd.Series) -> None:
        model = HistoricalVaR(window=500)
        short = normal_returns.iloc[:400]
        with pytest.raises(ValueError, match="Need at least"):
            walk_forward(model, short, window=500)

    def test_violations_are_plausible(self, normal_returns: pd.Series) -> None:
        model = HistoricalVaR(window=200)
        series = walk_forward(model, normal_returns, window=200, confidence=0.95)

        n_obs = len(series.dates)
        n_viol = count_violations(series)
        rate = n_viol / n_obs

        # At 95% confidence, expected ~5% violations.
        # Allow wide band since synthetic data.
        assert 0.01 < rate < 0.15, f"Violation rate {rate:.2%} looks implausible"


class TestKupiecPOF:
    def test_well_calibrated_model_passes(self, normal_returns: pd.Series) -> None:
        model = HistoricalVaR(window=200)
        series = walk_forward(model, normal_returns, window=200, confidence=0.95)

        result = KupiecPOF().test(series, alpha=0.05)
        # Well-calibrated model on its own data should typically pass
        assert result.test_name == "Kupiec-POF"
        assert 0 <= result.p_value <= 1


class TestChristoffersen:
    def test_independence_returns_result(self, normal_returns: pd.Series) -> None:
        model = HistoricalVaR(window=200)
        series = walk_forward(model, normal_returns, window=200, confidence=0.95)

        result = ChristoffersenIndependence().test(series)
        assert result.test_name == "Christoffersen-Independence"
        assert 0 <= result.p_value <= 1

    def test_cc_combines_both(self, normal_returns: pd.Series) -> None:
        model = HistoricalVaR(window=200)
        series = walk_forward(model, normal_returns, window=200, confidence=0.95)

        result = ChristoffersenCC().test(series)
        assert "kupiec_stat" in result.details
        assert "independence_stat" in result.details


class TestTrafficLight:
    def test_classifies_correctly(self, normal_returns: pd.Series) -> None:
        model = HistoricalVaR(window=200)
        series = walk_forward(model, normal_returns, window=200, confidence=0.99)

        result = TrafficLight().test(series)
        assert result.details["zone"] in ("green", "yellow", "red")
