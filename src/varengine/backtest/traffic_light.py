"""Basel Traffic Light test for VaR model adequacy.

The Basel Committee classifies models into three zones based on the
number of violations observed over 250 trading days at 99% confidence:

    Green  (0–4 violations):   model is acceptable
    Yellow (5–9 violations):   model is questionable, increased capital multiplier
    Red    (10+ violations):   model is rejected, mandatory capital surcharge

This is NOT a statistical test (no p-value) — it's a regulatory
classification.  We implement it as a BacktestMethod for interface
consistency, using the zone boundaries as a pseudo-test.

Reference: Basel Committee on Banking Supervision (1996, rev. 2006).
"Supervisory framework for the use of backtesting."
"""

from __future__ import annotations

from varengine.backtest.runner import count_violations
from varengine.protocols import BacktestResult, RollingSeries

# Default Basel thresholds for 250-day window at 99%
_GREEN_MAX = 4
_YELLOW_MAX = 9


class TrafficLight:
    """Basel Traffic Light classification."""

    def __init__(
        self,
        green_max: int = _GREEN_MAX,
        yellow_max: int = _YELLOW_MAX,
    ) -> None:
        self.green_max = green_max
        self.yellow_max = yellow_max

    def test(self, series: RollingSeries, alpha: float = 0.05) -> BacktestResult:
        n_violations = count_violations(series)
        n_obs = len(series.dates)

        if n_violations <= self.green_max:
            zone = "green"
            reject = False
        elif n_violations <= self.yellow_max:
            zone = "yellow"
            reject = False  # questionable but not rejected
        else:
            zone = "red"
            reject = True

        return BacktestResult(
            test_name=self.name,
            statistic=float(n_violations),
            p_value=0.0,  # not a statistical test
            reject_null=reject,
            alpha=alpha,
            details={
                "zone": zone,
                "n_violations": n_violations,
                "n_observations": n_obs,
                "thresholds": {
                    "green": f"0–{self.green_max}",
                    "yellow": f"{self.green_max + 1}–{self.yellow_max}",
                    "red": f"{self.yellow_max + 1}+",
                },
            },
        )

    @property
    def name(self) -> str:
        return "Basel-TrafficLight"
