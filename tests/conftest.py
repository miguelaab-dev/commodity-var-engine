"""Shared pytest fixtures — synthetic data so tests never hit the network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def normal_returns(rng: np.random.Generator) -> pd.Series:
    """1000 days of synthetic normal returns (μ=0.0002, σ=0.015)."""
    r = rng.normal(loc=0.0002, scale=0.015, size=1000)
    dates = pd.bdate_range("2020-01-01", periods=1000)
    return pd.Series(r, index=dates, name="portfolio")


@pytest.fixture
def fat_tail_returns(rng: np.random.Generator) -> pd.Series:
    """1000 days of synthetic t-distributed returns (fat tails, df=4)."""
    raw = rng.standard_t(df=4, size=1000)
    r = raw * 0.015 / np.sqrt(4 / (4 - 2))  # scale to ~same vol
    dates = pd.bdate_range("2020-01-01", periods=1000)
    return pd.Series(r, index=dates, name="portfolio")


@pytest.fixture
def multi_asset_returns(rng: np.random.Generator) -> pd.DataFrame:
    """1000 days of 3-asset correlated returns."""
    # Target correlation
    corr = np.array(
        [
            [1.0, 0.6, 0.2],
            [0.6, 1.0, 0.3],
            [0.2, 0.3, 1.0],
        ]
    )
    L = np.linalg.cholesky(corr)
    Z = rng.standard_normal((1000, 3))
    correlated = Z @ L.T * 0.015

    dates = pd.bdate_range("2020-01-01", periods=1000)
    return pd.DataFrame(
        correlated,
        index=dates,
        columns=["ASSET_A", "ASSET_B", "ASSET_C"],
    )


@pytest.fixture
def sample_prices(normal_returns: pd.Series) -> pd.DataFrame:
    """Synthetic price series derived from normal_returns."""
    prices = (1 + normal_returns).cumprod() * 100
    return prices.to_frame("CL=F")
