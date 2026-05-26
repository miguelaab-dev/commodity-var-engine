"""Portfolio construction and weighted return computation.

A portfolio here represents *notional-dollar-weighted* exposure to
commodity futures.  We deliberately avoid contract-count weighting
because it conflates notional size with price level.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


@dataclass
class Portfolio:
    """A commodity futures portfolio defined by its constituents and weights.

    Attributes
    ----------
    assets : dict mapping ticker → metadata (name, weight).
    """

    assets: dict[str, dict[str, object]]

    # -- Construction --------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> Portfolio:
        """Load portfolio definition from a YAML config file."""
        with open(path) as f:
            cfg = yaml.safe_load(f)

        assets = cfg["portfolio"]["assets"]
        instance = cls(assets=assets)
        instance._validate()
        return instance

    # -- Properties ----------------------------------------------------------

    @property
    def tickers(self) -> list[str]:
        return list(self.assets.keys())

    @property
    def weights(self) -> pd.Series:
        """Notional-dollar weights as a pandas Series (sums to 1)."""
        w = pd.Series({t: float(m["weight"]) for t, m in self.assets.items()})
        return w / w.sum()  # normalise to 1

    @property
    def names(self) -> dict[str, str]:
        return {t: str(m.get("name", t)) for t, m in self.assets.items()}

    # -- Return computation --------------------------------------------------

    def portfolio_returns(self, asset_returns: pd.DataFrame) -> pd.Series:
        """Compute the weighted portfolio return series.

        Parameters
        ----------
        asset_returns : DataFrame with one column per ticker (arithmetic returns).

        Returns
        -------
        Series of daily portfolio returns.
        """
        w = self.weights
        common = w.index.intersection(asset_returns.columns)
        if len(common) < len(w):
            missing = set(w.index) - set(common)
            raise ValueError(f"Return data missing for tickers: {missing}")

        return asset_returns[common].dot(w[common])

    def diversification_ratio(self, asset_returns: pd.DataFrame) -> float:
        """Ratio of weighted-average individual vol to portfolio vol.

        Values > 1 indicate diversification benefit.
        """
        w = self.weights.values
        cov = asset_returns[self.tickers].cov().values
        port_vol = np.sqrt(w @ cov @ w)
        individual_vols = np.sqrt(np.diag(cov))
        weighted_vol = w @ individual_vols
        return float(weighted_vol / port_vol)

    # -- Validation ----------------------------------------------------------

    def _validate(self) -> None:
        for ticker, meta in self.assets.items():
            if "weight" not in meta:
                raise ValueError(f"Ticker '{ticker}' missing 'weight' in config.")
            w = float(meta["weight"])
            if w < 0:
                raise ValueError(f"Negative weight for '{ticker}': {w}")
        total = sum(float(m["weight"]) for m in self.assets.values())
        if abs(total - 1.0) > 0.01:
            # warn but don't crash — weights are normalised at runtime
            import warnings
            warnings.warn(
                f"Portfolio weights sum to {total:.4f}, not 1.0. "
                "They will be normalised automatically.",
                stacklevel=2,
            )
