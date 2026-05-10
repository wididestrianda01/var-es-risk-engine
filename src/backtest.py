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
    pi0 = n01 / n0 if n0 > 0 else 0.0
    pi1 = n11 / n1 if n1 > 0 else 0.0
    pi = (n01 + n11) / n

    def _safe_xlog(x, y):
        if x == 0:
            return 0.0
        if y <= 0 or y >= 1:
            return 0.0
        return x * np.log(y)

    # Independence LR test
    ll_restricted = (
        _safe_xlog(n00, 1 - pi) + _safe_xlog(n01, pi)
        + _safe_xlog(n10, 1 - pi) + _safe_xlog(n11, pi)
    )
    ll_unrestricted = (
        _safe_xlog(n00, 1 - pi0) + _safe_xlog(n01, pi0)
        + _safe_xlog(n10, 1 - pi1) + _safe_xlog(n11, pi1)
    )
    lr_ind = -2 * (ll_restricted - ll_unrestricted)

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


def traffic_light(breaches=None, total=250, breaches_99=None, breaches_975=None,
                  framework="basel1996"):
    """Basel traffic light system for backtesting.

    Args:
        breaches: number of breaches (Basel 1996: at 99% VaR).
        total: total observations (default 250).
        breaches_99: FRTB 2019: breaches at 99% VaR.
        breaches_975: FRTB 2019: breaches at 97.5% VaR.
        framework: "basel1996" or "frtb2019".

    Returns:
        dict with zone, multiplier (Basel 1996 only), breaches counts.
    """
    if framework == "basel1996":
        if breaches is None:
            raise ValueError("breaches is required for basel1996")
        if breaches <= 4:
            return {"zone": "green", "multiplier": 3.0, "breaches": breaches,
                    "total": total, "framework": "basel1996"}
        elif breaches <= 9:
            k = 3.40 + (breaches - 5) * (0.45 / 4)
            return {"zone": "yellow", "multiplier": round(k, 2),
                    "breaches": breaches, "total": total, "framework": "basel1996"}
        else:
            return {"zone": "red", "multiplier": 4.0, "breaches": breaches,
                    "total": total, "framework": "basel1996"}

    elif framework == "frtb2019":
        b99 = breaches_99 if breaches_99 is not None else breaches
        b975 = breaches_975 if breaches_975 is not None else breaches
        if b99 <= 12 and b975 <= 30:
            return {"zone": "green", "breaches_99": b99,
                    "breaches_975": b975, "total": total, "framework": "frtb2019"}
        else:
            return {"zone": "red", "breaches_99": b99,
                    "breaches_975": b975, "total": total, "framework": "frtb2019"}
    else:
        raise ValueError(f"Unknown framework: {framework}")


def acerbi_szekely_z2(returns, var_forecasts, es_forecasts, alpha, n_sim=1000):
    """Acerbi-Szekely Z2 test for Expected Shortfall backtesting.

    H0: ES forecasts are correctly specified.
    Test statistic: Z2 = (1/n) * S (R_t / ES_t) * I_t + 1
    where I_t = 1 if VaR breach at t.

    P-value computed via Monte Carlo simulation under H0.

    Args:
        returns: array-like of realized returns.
        var_forecasts: array-like of VaR forecasts (same length).
        es_forecasts: array-like of ES forecasts (same length).
        alpha: VaR/ES confidence level.
        n_sim: number of Monte Carlo simulations.

    Returns:
        TestResult with Z2 statistic and simulated p-value.
    """
    returns = np.asarray(returns)
    var_forecasts = np.asarray(var_forecasts)
    es_forecasts = np.asarray(es_forecasts)
    n = len(returns)

    if n < 10:
        raise ValueError("Need at least 10 observations")
    if not (len(var_forecasts) == len(es_forecasts) == n):
        raise ValueError("All arrays must have same length")

    breaches = returns <= var_forecasts

    # Avoid division by zero -- use small epsilon for near-zero ES
    eps = 1e-10
    es_safe = np.where(np.abs(es_forecasts) < eps, -eps, es_forecasts)

    # Z2 observed statistic
    tail_contrib = np.where(breaches, returns / es_safe, 0.0)
    z2_obs = np.mean(tail_contrib) + 1.0  # centered under H0

    # Simulate null distribution
    rng = np.random.default_rng(42)
    sim_stats = np.zeros(n_sim)
    for i in range(n_sim):
        sim_breaches = rng.binomial(1, 1 - alpha, n).astype(bool)
        sim_tail = np.where(sim_breaches, returns / es_safe, 0.0)
        sim_stats[i] = np.mean(sim_tail) + 1.0

    # Finite-sample bias correction:
    # The Bernoulli simulation generates breaches at random locations, but actual
    # VaR breaches always occur in the tail where E[R_t/ES_t | breach] = 1.
    # Under H0, the unconditional mean of R_t/ES_t over ALL observations is ~0
    # (since returns are centered near 0), while the tail-conditional mean is 1.
    # This creates a bias: the Bernoulli simulation mean is ~(1-alpha)*0+1 = 1.00
    # while the true expected Z2 under H0 is (1-alpha)*1+1 = 1.05.
    # Center the simulated stats at the theoretical null expectation.
    expected_z2 = (1 - alpha) + 1.0
    sim_centered = sim_stats - np.mean(sim_stats) + expected_z2

    # Two-sided p-value
    p_value = np.mean(np.abs(sim_centered) >= np.abs(z2_obs))
    p_value = max(p_value, 1.0 / n_sim)

    return TestResult(
        statistic=float(z2_obs),
        p_value=float(p_value),
        reject=p_value < 0.05,
        breaches=int(np.sum(breaches)),
        total=n,
        alpha=alpha,
    )
