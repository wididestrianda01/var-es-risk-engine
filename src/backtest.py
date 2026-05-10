from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class TestResult:
    statistic: float
    p_value: float
    reject: bool
    breaches: int
    total: int
    alpha: float


def kupiec_test(
    breaches: int, total: int, alpha: float
) -> TestResult:
    """Kupiec POF test — binomial test on VaR breach frequency.

    H0: breach rate = 1 - alpha.

    Args:
        breaches: number of VaR breaches observed.
        total: total number of observations.
        alpha: VaR confidence level.

    Returns:
        TestResult with test statistic and p-value.
    """
    if breaches > total:
        raise ValueError(f"breaches ({breaches}) > total ({total})")
    if breaches < 0:
        raise ValueError("breaches must be >= 0")
    if total <= 0:
        raise ValueError("total must be > 0")

    expected_p = 1 - alpha
    observed_p = breaches / total

    if breaches == 0:
        lr_stat = -2 * total * np.log(1 - expected_p)
    elif breaches == total:
        lr_stat = -2 * total * np.log(expected_p)
    else:
        lr_restricted = stats.binom.logpmf(breaches, total, expected_p)
        lr_unrestricted = stats.binom.logpmf(breaches, total, observed_p)
        lr_stat = -2 * (lr_restricted - lr_unrestricted)

    lr_stat = max(lr_stat, 0.0)
    p_value = 1 - stats.chi2.cdf(lr_stat, df=1)

    return TestResult(
        statistic=float(lr_stat),
        p_value=float(p_value),
        reject=p_value < 0.05,
        breaches=breaches,
        total=total,
        alpha=alpha,
    )
