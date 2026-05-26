"""Download and cache commodity futures data from Yahoo Finance.

Key design decisions:
- Uses ARITHMETIC returns (not log) because WTI went negative in Apr 2020.
- Detects and flags roll-date return spikes in continuous futures contracts.
- Caches raw data to data/ so repeated runs don't hit the network.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# yfinance is imported lazily to keep test imports fast
_CACHE_DIR = Path("data")


def download_prices(
    tickers: list[str],
    start: str = "2010-01-01",
    end: str | None = None,
    cache_dir: Path = _CACHE_DIR,
) -> pd.DataFrame:
    """Download adjusted close prices for *tickers* and cache as CSV.

    Returns a DataFrame indexed by date with one column per ticker.
    Missing dates are forward-filled (holidays differ across exchanges).
    """
    import yfinance as yf  # lazy import

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "raw_prices.csv"

    if cache_path.exists():
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        # Only re-download if tickers changed
        if set(tickers).issubset(df.columns):
            return df[tickers]

    raw: pd.DataFrame = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )["Close"]

    if isinstance(raw, pd.Series):
        raw = raw.to_frame(tickers[0])

    raw = raw.ffill().dropna()
    raw.to_csv(cache_path)
    return raw


def compute_returns(
    prices: pd.DataFrame,
    method: str = "arithmetic",
) -> pd.DataFrame:
    """Compute daily returns from a price DataFrame.

    Parameters
    ----------
    method : 'arithmetic' (default, recommended) or 'log'.
        Arithmetic is used because commodity futures can have zero or
        negative prices (WTI, Apr 2020).  Log returns are undefined in
        that regime.
    """
    if method == "arithmetic":
        returns = prices.pct_change().dropna()
    elif method == "log":
        if (prices <= 0).any().any():
            raise ValueError(
                "Log returns requested but prices contain zero/negative values. "
                "Use method='arithmetic' for commodity futures."
            )
        returns = np.log(prices / prices.shift(1)).dropna()
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'arithmetic' or 'log'.")

    return returns


def flag_roll_dates(
    returns: pd.DataFrame,
    z_threshold: float = 6.0,
) -> pd.DataFrame:
    """Return a boolean mask of probable futures roll dates.

    Roll-date returns show as extreme outliers caused by the price gap
    between the expiring and next-month contract — NOT real market moves.
    We flag (but don't auto-remove) any single-day return exceeding
    *z_threshold* standard deviations from the mean.
    """
    z_scores = (returns - returns.mean()) / returns.std()
    return z_scores.abs() > z_threshold


def clean_returns(
    returns: pd.DataFrame,
    z_threshold: float = 6.0,
) -> pd.DataFrame:
    """Remove probable roll-date spikes from the return series."""
    flags = flag_roll_dates(returns, z_threshold=z_threshold)
    cleaned = returns.copy()
    cleaned[flags] = np.nan
    return cleaned.dropna()
