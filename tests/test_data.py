"""Tests for data loading and portfolio construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from varengine.data.loader import compute_returns, flag_roll_dates
from varengine.data.portfolio import Portfolio


class TestComputeReturns:
    def test_arithmetic_returns(self) -> None:
        prices = pd.DataFrame({"A": [100.0, 105.0, 110.0]})
        r = compute_returns(prices, method="arithmetic")

        expected = pd.DataFrame({"A": [0.05, 110 / 105 - 1]}, index=[1, 2])
        pd.testing.assert_frame_equal(r.reset_index(drop=True), expected.reset_index(drop=True), atol=1e-10)

    def test_log_returns_reject_negative_prices(self) -> None:
        prices = pd.DataFrame({"A": [100.0, -37.63, 20.0]})
        with pytest.raises(ValueError, match="zero/negative"):
            compute_returns(prices, method="log")

    def test_unknown_method_raises(self) -> None:
        prices = pd.DataFrame({"A": [100.0, 105.0]})
        with pytest.raises(ValueError, match="Unknown method"):
            compute_returns(prices, method="magic")


class TestFlagRollDates:
    def test_flags_extreme_spike(self, rng: np.random.Generator) -> None:
        normal = rng.normal(0, 0.01, 500)
        normal[250] = 0.20  # inject a roll-date spike
        r = pd.DataFrame({"A": normal})

        flags = flag_roll_dates(r, z_threshold=6.0)
        assert flags.iloc[250]["A"], "Should flag the injected spike"
        assert flags.sum().sum() >= 1


class TestPortfolio:
    def test_from_yaml(self, tmp_path) -> None:
        config = tmp_path / "test.yaml"
        config.write_text("""
portfolio:
  assets:
    A:
      name: Asset A
      weight: 0.6
    B:
      name: Asset B
      weight: 0.4
""")
        p = Portfolio.from_yaml(config)
        assert p.tickers == ["A", "B"]
        np.testing.assert_allclose(p.weights.values, [0.6, 0.4])

    def test_weights_normalise(self, tmp_path) -> None:
        config = tmp_path / "test.yaml"
        config.write_text("""
portfolio:
  assets:
    A: { name: A, weight: 2 }
    B: { name: B, weight: 3 }
""")
        p = Portfolio.from_yaml(config)
        np.testing.assert_allclose(p.weights.sum(), 1.0)

    def test_portfolio_returns(self, multi_asset_returns: pd.DataFrame) -> None:
        p = Portfolio(assets={
            "ASSET_A": {"name": "A", "weight": 0.5},
            "ASSET_B": {"name": "B", "weight": 0.3},
            "ASSET_C": {"name": "C", "weight": 0.2},
        })
        port_ret = p.portfolio_returns(multi_asset_returns)
        assert len(port_ret) == len(multi_asset_returns)
        assert isinstance(port_ret, pd.Series)

    def test_missing_ticker_raises(self) -> None:
        p = Portfolio(assets={"MISSING": {"name": "X", "weight": 1.0}})
        returns = pd.DataFrame({"OTHER": [0.01, 0.02]})
        with pytest.raises(ValueError, match="missing for tickers"):
            p.portfolio_returns(returns)
