"""Plotly-based visualisations with static PNG export for README.

All functions return a plotly Figure AND optionally save a static PNG
to reports/ for embedding in the README (since GitHub doesn't render
interactive Plotly in notebooks).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from varengine.protocols import RiskEstimate, RollingSeries

_REPORTS = Path("reports")


def _save_static(fig: go.Figure, filename: str) -> None:
    """Save figure as PNG for README embedding."""
    _REPORTS.mkdir(exist_ok=True)
    fig.write_image(_REPORTS / f"{filename}.png", width=1200, height=600, scale=2)


def plot_var_comparison(
    estimates: list[RiskEstimate],
    title: str = "VaR & ES Comparison",
    save: bool = True,
) -> go.Figure:
    """Bar chart comparing VaR and ES across methods."""
    methods = [e.method for e in estimates]
    vars_ = [e.var for e in estimates]
    ess = [e.es for e in estimates]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="VaR", x=methods, y=vars_, marker_color="#2563eb"))
    fig.add_trace(go.Bar(name="ES", x=methods, y=ess, marker_color="#dc2626"))
    fig.update_layout(
        title=title,
        yaxis_title="Loss threshold",
        barmode="group",
        template="plotly_white",
    )

    if save:
        _save_static(fig, "var_comparison")
    return fig


def plot_backtest(
    series: RollingSeries,
    title: str | None = None,
    save: bool = True,
) -> go.Figure:
    """Time series of returns with VaR breach highlighting."""
    losses = -series.realised
    breaches = losses > series.var_series

    fig = make_subplots(rows=1, cols=1)

    # Returns
    fig.add_trace(go.Scatter(
        x=series.dates, y=series.realised,
        mode="lines", name="Return", line=dict(color="#6b7280", width=0.8),
    ))

    # VaR line (inverted to return space)
    fig.add_trace(go.Scatter(
        x=series.dates, y=-series.var_series,
        mode="lines", name=f"VaR ({series.confidence:.0%})",
        line=dict(color="#2563eb", width=1.5, dash="dash"),
    ))

    # Breach markers
    breach_dates = series.dates[breaches]
    breach_returns = series.realised[breaches]
    fig.add_trace(go.Scatter(
        x=breach_dates, y=breach_returns,
        mode="markers", name="Violation",
        marker=dict(color="#dc2626", size=6, symbol="x"),
    ))

    fig.update_layout(
        title=title or f"Backtest: {series.method}",
        yaxis_title="Daily return",
        template="plotly_white",
        showlegend=True,
    )

    if save:
        _save_static(fig, f"backtest_{series.method.lower().replace(' ', '_')}")
    return fig


def plot_correlation_heatmap(
    returns: pd.DataFrame,
    save: bool = True,
) -> go.Figure:
    """Correlation matrix heatmap for portfolio constituents."""
    corr = returns.corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        text=corr.round(2).values,
        texttemplate="%{text}",
    ))
    fig.update_layout(
        title="Commodity Return Correlations",
        template="plotly_white",
    )

    if save:
        _save_static(fig, "correlation_heatmap")
    return fig
