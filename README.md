# commodity-var-engine

[![CI](https://github.com/miguellaab/commodity-var-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/miguellaab/commodity-var-engine/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

VaR and Expected Shortfall engine for commodity futures portfolios. Implements five estimation methods, three volatility models, and a complete walk-forward backtesting framework with Kupiec, Christoffersen, and Basel Traffic Light tests.

## Why This Exists

Calculating VaR without backtesting is just drawing a pretty chart. This engine implements the full pipeline: data → estimation → backtesting → visualisation, with composable architecture so any volatility model plugs into any simulation framework.

## Quick Start

```bash
git clone git@github.com:miguellaab/commodity-var-engine.git
cd commodity-var-engine

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Compute VaR for the default portfolio
varengine run --config configs/portfolio.yaml

# Run walk-forward backtest
varengine backtest --config configs/portfolio.yaml --window 500
```

## VaR Methods

| Method | Distributional Assumption | Volatility | Handles Fat Tails? |
|--------|--------------------------|------------|-------------------|
| Historical Simulation | None (empirical) | Implicit | ✓ |
| Parametric | Normal | Sample / EWMA | ✗ |
| Monte Carlo | Normal or Student-t | Constant / EWMA / GARCH | ✓ (with t + GARCH) |
| Cornish-Fisher | CF expansion | Sample | Partially |

All methods compute both VaR and Expected Shortfall (ES).

## Architecture

```
varengine/
├── data/          # Download, cache, portfolio construction
├── models/
│   └── volatility/  # Constant, EWMA, GARCH(1,1)
├── var/           # Historical, Parametric, Monte Carlo, Cornish-Fisher
├── backtest/      # Walk-forward runner + Kupiec, Christoffersen, Traffic Light
├── viz/           # Plotly charts + static PNG export
└── cli.py         # Command-line interface
```

**Key design decisions:**
- **Arithmetic returns** (not log) — WTI went negative in April 2020
- **Composable volatility** — GARCH is a vol model, not a distribution; plug any vol model into Monte Carlo
- **Walk-forward only** — in-sample backtesting is not supported because it's meaningless
- **Protocol-driven** — all models implement `VaRModel` protocol; all tests implement `BacktestMethod`

## Portfolio

Default portfolio (configurable via `configs/portfolio.yaml`):

| Commodity | Ticker | Weight | Sector |
|-----------|--------|--------|--------|
| WTI Crude | CL=F | 20% | Energy |
| Natural Gas | NG=F | 10% | Energy |
| Gold | GC=F | 20% | Metals |
| Copper | HG=F | 15% | Metals |
| Soybeans | ZS=F | 15% | Agriculture |
| Corn | ZC=F | 10% | Agriculture |
| Coffee | KC=F | 10% | Soft |

## Documentation

- [`docs/var_methods.md`](docs/var_methods.md) — VaR and ES methodology with formulas and references
- [`docs/backtesting.md`](docs/backtesting.md) — Backtesting framework and test interpretation

## Development

```bash
make dev          # Install with dev dependencies
make test         # Run all tests with coverage
make test-fast    # Skip slow tests (network, GARCH fitting)
make lint         # Ruff linting
make type-check   # mypy strict mode
make format       # Auto-format
```

## License

MIT
