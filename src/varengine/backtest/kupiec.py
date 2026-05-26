"""Kupiec Proportion of Failures (POF) test.

Tests whether the observed violation rate equals the expected rate
(1 − confidence).  Under H0 (model is correctly specified), the
number of violations follows Binomial(n, 1−α).

The test statistic is a likelihood ratio:

    LR_POF = −2 ln[ (1−p)^(n−x) · p^x / ((1−p̂)^(n−x) · p̂^x) ]

where p = 1−α (expected rate), p̂ = x/n (observed rate), x = violations.

LR_POF ~ χ²(1) under H0.

Reference: Kupiec, P. (1995). "Techniques for Verifying the Accuracy
of Risk Measurement Models." Journal of Derivatives.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from varengine.backtest.runner import count_violations
from varengine.protocols import BacktestResult, RollingSeries


class KupiecPOF:
    """Kupiec Proportion of Failures test."""

    def test(self, series: RollingSeries, alpha: float = 0.05) -> BacktestResult:
        n = len(series.dates)
        x = count_violations(series)
        p = 1 - series.confidence  # expected violation rate

        if x == 0:
            # No violations: LR is well-defined but model is likely too conservative
            lr_stat = -2 * n * np.log(1 - p)
        elif x == n:
            lr_stat = -2 * n * np.log(p)
        else:
            p_hat = x / n
            lr_stat = -2 * ((n - x) * np.log((1 - p) / (1 - p_hat)) + x * np.log(p / p_hat))

        p_value = 1 - sp_stats.chi2.cdf(lr_stat, df=1)

        return BacktestResult(
            test_name=self.name,
            statistic=float(lr_stat),
            p_value=float(p_value),
            reject_null=p_value < alpha,
            alpha=alpha,
            details={
                "n_observations": n,
                "n_violations": x,
                "expected_rate": p,
                "observed_rate": x / n if n > 0 else 0.0,
            },
        )

    @property
    def name(self) -> str:
        return "Kupiec-POF"
