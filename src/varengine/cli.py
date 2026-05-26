"""Command-line interface for the VaR engine.

Usage:
    varengine run --config configs/portfolio.yaml
    varengine run --config configs/portfolio.yaml --confidence 0.95
    varengine backtest --config configs/portfolio.yaml --window 750
"""

from __future__ import annotations

from pathlib import Path

import click
import pandas as pd


@click.group()
def main() -> None:
    """Commodity VaR Engine — compute VaR, ES, and run backtests."""


@main.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="YAML config.")
@click.option("--confidence", default=0.99, type=float, help="Confidence level (default 0.99).")
@click.option("--start", default="2010-01-01", help="Data start date.")
def run(config: str, confidence: float, start: str) -> None:
    """Compute VaR and ES for all methods."""
    from varengine.data.loader import clean_returns, compute_returns, download_prices
    from varengine.data.portfolio import Portfolio
    from varengine.models.volatility.ewma import EWMAVolatility
    from varengine.var import CornishFisherVaR, HistoricalVaR, MonteCarloVaR, ParametricVaR

    portfolio = Portfolio.from_yaml(config)
    click.echo(f"Portfolio: {len(portfolio.tickers)} assets")

    prices = download_prices(portfolio.tickers, start=start)
    returns = clean_returns(compute_returns(prices, method="arithmetic"))
    port_returns = portfolio.portfolio_returns(returns)

    click.echo(f"Data: {len(port_returns)} observations ({port_returns.index[0].date()} → {port_returns.index[-1].date()})")
    click.echo(f"Confidence: {confidence:.1%}\n")

    models = [
        HistoricalVaR(window=500),
        ParametricVaR(),
        ParametricVaR(vol_model=EWMAVolatility()),
        MonteCarloVaR(dist="normal"),
        MonteCarloVaR(dist="t", df=5),
        CornishFisherVaR(),
    ]

    click.echo(f"{'Method':<40} {'VaR':>10} {'ES':>10}")
    click.echo("─" * 62)

    for model in models:
        model.fit(port_returns)
        est = model.estimate(confidence=confidence)
        click.echo(f"{est.method:<40} {est.var:>10.6f} {est.es:>10.6f}")


@main.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
@click.option("--confidence", default=0.99, type=float)
@click.option("--window", "-w", default=500, type=int, help="Rolling window for backtest.")
@click.option("--start", default="2010-01-01")
def backtest(config: str, confidence: float, window: int, start: str) -> None:
    """Run walk-forward backtest with all statistical tests."""
    from varengine.backtest.christoffersen import ChristoffersenCC, ChristoffersenIndependence
    from varengine.backtest.kupiec import KupiecPOF
    from varengine.backtest.runner import count_violations, walk_forward
    from varengine.backtest.traffic_light import TrafficLight
    from varengine.data.loader import clean_returns, compute_returns, download_prices
    from varengine.data.portfolio import Portfolio
    from varengine.var import HistoricalVaR

    portfolio = Portfolio.from_yaml(config)
    prices = download_prices(portfolio.tickers, start=start)
    returns = clean_returns(compute_returns(prices, method="arithmetic"))
    port_returns = portfolio.portfolio_returns(returns)

    model = HistoricalVaR(window=window)
    click.echo(f"Running walk-forward backtest ({model.name})...")

    series = walk_forward(model, port_returns, window=window, confidence=confidence)

    n_violations = count_violations(series)
    click.echo(f"Violations: {n_violations} / {len(series.dates)} ({n_violations/len(series.dates):.2%})")
    click.echo(f"Expected:   {1 - confidence:.2%}\n")

    tests = [KupiecPOF(), ChristoffersenIndependence(), ChristoffersenCC(), TrafficLight()]

    click.echo(f"{'Test':<30} {'Statistic':>10} {'p-value':>10} {'Result':>10}")
    click.echo("─" * 62)

    for t in tests:
        result = t.test(series)
        status = "REJECT" if result.reject_null else "PASS"
        p_str = f"{result.p_value:.4f}" if result.p_value > 0 else "N/A"
        zone = result.details.get("zone", "")
        extra = f" [{zone}]" if zone else ""
        click.echo(f"{result.test_name:<30} {result.statistic:>10.4f} {p_str:>10} {status:>10}{extra}")


if __name__ == "__main__":
    main()
