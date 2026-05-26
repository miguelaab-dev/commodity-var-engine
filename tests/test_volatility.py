"""Tests for volatility models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from varengine.models.volatility.constant import ConstantVolatility
from varengine.models.volatility.ewma import EWMAVolatility


class TestConstantVol:
    def test_matches_sample_std(self, normal_returns: pd.Series) -> None:
        model = ConstantVolatility()
        model.fit(normal_returns)
        forecast = model.forecast(horizon=1)

        expected = normal_returns.std(ddof=1)
        np.testing.assert_allclose(forecast[0], expected, rtol=1e-10)

    def test_forecast_is_flat(self, normal_returns: pd.Series) -> None:
        model = ConstantVolatility()
        model.fit(normal_returns)
        forecast = model.forecast(horizon=10)

        assert len(forecast) == 10
        assert np.all(forecast == forecast[0])


class TestEWMA:
    def test_forecast_positive(self, normal_returns: pd.Series) -> None:
        model = EWMAVolatility(lam=0.94)
        model.fit(normal_returns)
        forecast = model.forecast(horizon=5)

        assert np.all(forecast > 0)
        assert len(forecast) == 5

    def test_reacts_to_shock(self, rng: np.random.Generator) -> None:
        # Calm period followed by a shock
        calm = rng.normal(0, 0.01, 100)
        shock = np.array([0.10])  # 10% daily move
        post = rng.normal(0, 0.01, 10)
        r = pd.Series(np.concatenate([calm, shock, post]))

        model = EWMAVolatility(lam=0.94)
        model.fit(r)
        vol = model.conditional_vol

        # Vol after shock should be elevated
        assert vol[101] > vol[99], "EWMA should spike after a shock"

    def test_invalid_lambda(self) -> None:
        with pytest.raises(ValueError):
            EWMAVolatility(lam=1.5)
