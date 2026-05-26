"""Christoffersen Independence and Conditional Coverage tests.

Kupiec tests whether the violation *rate* is correct but ignores
clustering.  A model that produces violations in bursts (e.g. 5
consecutive breach days) can pass Kupiec but is still badly specified.

The Independence test checks whether violations are serially
independent by fitting a first-order Markov chain to the violation
sequence and testing whether transition probabilities are equal.

The Conditional Coverage test combines Kupiec + Independence into
a single joint test with 2 degrees of freedom.

Reference: Christoffersen, P. (1998). "Evaluating Interval Forecasts."
International Economic Review.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from varengine.protocols import BacktestResult, RollingSeries


def _transition_counts(series: RollingSeries) -> tuple[int, int, int, int]:
    """Count transitions in the violation indicator sequence.

    Returns (n00, n01, n10, n11) where:
        nij = number of times state i is followed by state j.
        State 0 = no violation, State 1 = violation.
    """
    losses = -series.realised.values
    var_vals = series.var_series.values
    hits = (losses > var_vals).astype(int)

    n00 = n01 = n10 = n11 = 0
    for t in range(len(hits) - 1):
        i, j = hits[t], hits[t + 1]
        if i == 0 and j == 0:
            n00 += 1
        elif i == 0 and j == 1:
            n01 += 1
        elif i == 1 and j == 0:
            n10 += 1
        else:
            n11 += 1

    return n00, n01, n10, n11


class ChristoffersenIndependence:
    """Markov independence test for VaR violation clustering."""

    def test(self, series: RollingSeries, alpha: float = 0.05) -> BacktestResult:
        n00, n01, n10, n11 = _transition_counts(series)

        n0 = n00 + n01  # total days starting in state 0
        n1 = n10 + n11  # total days starting in state 1
        n = n0 + n1

        if n0 == 0 or n1 == 0 or n01 + n11 == 0:
            # Degenerate case: cannot estimate transition probs
            return BacktestResult(
                test_name=self.name,
                statistic=0.0,
                p_value=1.0,
                reject_null=False,
                alpha=alpha,
                details={"degenerate": True},
            )

        # Unconditional violation probability
        p_hat = (n01 + n11) / n

        # Conditional probabilities
        p01 = n01 / n0 if n0 > 0 else 0
        p11 = n11 / n1 if n1 > 0 else 0

        # Likelihood ratio
        # L0 (independence): all transitions use p_hat
        # L1 (Markov): transitions use p01, p11
        log_l0 = 0.0
        log_l1 = 0.0

        if n00 > 0:
            log_l0 += n00 * np.log(1 - p_hat)
            log_l1 += n00 * np.log(1 - p01) if p01 < 1 else 0
        if n01 > 0:
            log_l0 += n01 * np.log(p_hat)
            log_l1 += n01 * np.log(p01) if p01 > 0 else 0
        if n10 > 0:
            log_l0 += n10 * np.log(1 - p_hat)
            log_l1 += n10 * np.log(1 - p11) if p11 < 1 else 0
        if n11 > 0:
            log_l0 += n11 * np.log(p_hat)
            log_l1 += n11 * np.log(p11) if p11 > 0 else 0

        lr_ind = -2 * (log_l0 - log_l1)
        lr_ind = max(lr_ind, 0.0)  # numerical floor

        p_value = 1 - sp_stats.chi2.cdf(lr_ind, df=1)

        return BacktestResult(
            test_name=self.name,
            statistic=float(lr_ind),
            p_value=float(p_value),
            reject_null=p_value < alpha,
            alpha=alpha,
            details={
                "n00": n00,
                "n01": n01,
                "n10": n10,
                "n11": n11,
                "p01": p01,
                "p11": p11,
            },
        )

    @property
    def name(self) -> str:
        return "Christoffersen-Independence"


class ChristoffersenCC:
    """Conditional Coverage = Kupiec POF + Independence (joint test, df=2)."""

    def test(self, series: RollingSeries, alpha: float = 0.05) -> BacktestResult:
        from varengine.backtest.kupiec import KupiecPOF

        kupiec_result = KupiecPOF().test(series, alpha=alpha)
        ind_result = ChristoffersenIndependence().test(series, alpha=alpha)

        lr_cc = kupiec_result.statistic + ind_result.statistic
        p_value = 1 - sp_stats.chi2.cdf(lr_cc, df=2)

        return BacktestResult(
            test_name=self.name,
            statistic=float(lr_cc),
            p_value=float(p_value),
            reject_null=p_value < alpha,
            alpha=alpha,
            details={
                "kupiec_stat": kupiec_result.statistic,
                "kupiec_pval": kupiec_result.p_value,
                "independence_stat": ind_result.statistic,
                "independence_pval": ind_result.p_value,
            },
        )

    @property
    def name(self) -> str:
        return "Christoffersen-CC"
