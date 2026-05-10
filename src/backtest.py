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


def christoffersen_test(breaches):
    """Christoffersen conditional coverage test.

    Tests whether VaR breaches are independent (not clustered).
    Uses a Markov chain model: P(breach_t | breach_{t-1}).

    Args:
        breaches: array-like of 0/1 breach indicators.

    Returns:
        TestResult with LR statistic and p-value.
    """
    breaches = np.asarray(breaches, dtype=int)
    n = len(breaches)
    if n < 2:
        raise ValueError("Need at least 2 observations")

    # Count transitions
    n00 = n01 = n10 = n11 = 0
    for t in range(1, n):
        if breaches[t - 1] == 0 and breaches[t] == 0:
            n00 += 1
        elif breaches[t - 1] == 0 and breaches[t] == 1:
            n01 += 1
        elif breaches[t - 1] == 1 and breaches[t] == 0:
            n10 += 1
        else:
            n11 += 1

    n0 = n00 + n01
    n1 = n10 + n11

    if n0 == 0 or n1 == 0:
        return TestResult(
            statistic=0.0, p_value=1.0, reject=False,
            breaches=int(np.sum(breaches)), total=n, alpha=np.nan,
        )

    # Unrestricted probabilities
    pi0 = n01 / n0
    pi1 = n11 / n1
    pi = (n01 + n11) / n

    # Independence LR test
    # LR = -2 * ln(L(restricted) / L(unrestricted))
    lr_ind = -2 * (
        (n00 * np.log(1 - pi) + n01 * np.log(pi)
         + n10 * np.log(1 - pi) + n11 * np.log(pi))
        - (n00 * np.log(1 - pi0) + n01 * np.log(pi0)
           + n10 * np.log(1 - pi1) + n11 * np.log(pi1))
    )

    lr_ind = max(lr_ind, 0)
    p_value = 1 - stats.chi2.cdf(lr_ind, df=1)

    return TestResult(
        statistic=float(lr_ind),
        p_value=float(p_value),
        reject=p_value < 0.05,
        breaches=int(np.sum(breaches)),
        total=n,
        alpha=np.nan,
    )
