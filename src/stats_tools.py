"""Small, dependency-light statistics helpers for A/B tests.

Written with NumPy and SciPy only, so every formula is visible. Each function is
checked against SciPy or a simulation in tests/test_stats_tools.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------- #
# Sample ratio mismatch
# --------------------------------------------------------------------------- #
def srm_test(counts: list[int], expected_share: list[float] | None = None) -> dict:
    """Chi-square goodness-of-fit test that traffic was split as designed."""
    counts = np.asarray(counts, dtype=float)
    if expected_share is None:
        expected_share = np.full(len(counts), 1 / len(counts))
    expected = counts.sum() * np.asarray(expected_share)
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    p = float(stats.chi2.sf(chi2, df=len(counts) - 1))
    return {"chi2": chi2, "p_value": p, "observed_share": (counts / counts.sum()).tolist()}


# --------------------------------------------------------------------------- #
# Difference in two proportions
# --------------------------------------------------------------------------- #
@dataclass
class PropTest:
    p_a: float
    p_b: float
    diff: float            # p_b - p_a, in proportion units
    ci_low: float
    ci_high: float
    rel_lift: float        # diff / p_a
    rel_ci_low: float
    rel_ci_high: float
    z: float
    p_value: float


def two_prop_test(x_a: int, n_a: int, x_b: int, n_b: int, alpha: float = 0.05) -> PropTest:
    """Two-sided z-test for p_b - p_a.

    The test statistic uses the pooled proportion (the null says the rates are
    equal); the confidence interval uses the unpooled standard error.
    The relative-lift interval uses the delta method on log(p_b / p_a).
    """
    p_a, p_b = x_a / n_a, x_b / n_b
    diff = p_b - p_a

    pooled = (x_a + x_b) / (n_a + n_b)
    se_pooled = np.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    z = diff / se_pooled
    p_value = 2 * stats.norm.sf(abs(z))

    zc = stats.norm.ppf(1 - alpha / 2)
    se = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)

    log_rr = np.log(p_b / p_a)
    se_log = np.sqrt((1 - p_a) / x_a + (1 - p_b) / x_b)
    return PropTest(
        p_a=p_a, p_b=p_b, diff=diff,
        ci_low=diff - zc * se, ci_high=diff + zc * se,
        rel_lift=p_b / p_a - 1,
        rel_ci_low=np.exp(log_rr - zc * se_log) - 1,
        rel_ci_high=np.exp(log_rr + zc * se_log) - 1,
        z=float(z), p_value=float(p_value),
    )


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values (controls family-wise error)."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted, running = {}, 0.0
    for i, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[name] = running
    return adjusted


# --------------------------------------------------------------------------- #
# Resampling and Bayesian views
# --------------------------------------------------------------------------- #
def bootstrap_prop_diff(x_a: int, n_a: int, x_b: int, n_b: int,
                        n_boot: int = 20_000, seed: int = 42) -> np.ndarray:
    """Bootstrap distribution of p_b - p_a.

    Resampling n 0/1 outcomes with replacement gives a Binomial(n, p_hat) count,
    so drawing binomials is the exact bootstrap for a proportion, just faster.
    """
    rng = np.random.default_rng(seed)
    a = rng.binomial(n_a, x_a / n_a, n_boot) / n_a
    b = rng.binomial(n_b, x_b / n_b, n_boot) / n_b
    return b - a


def bootstrap_mean_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 5_000,
                        seed: int = 42, chunk: int = 250) -> np.ndarray:
    """Bootstrap distribution of mean(b) - mean(a), resampling in chunks to save memory."""
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for start in range(0, n_boot, chunk):
        k = min(chunk, n_boot - start)
        ma = a[rng.integers(0, len(a), (k, len(a)))].mean(axis=1)
        mb = b[rng.integers(0, len(b), (k, len(b)))].mean(axis=1)
        out[start:start + k] = mb - ma
    return out


def beta_posterior(x_a: int, n_a: int, x_b: int, n_b: int,
                   draws: int = 200_000, seed: int = 42) -> dict:
    """Beta(1, 1) priors. P(B beats A) and the expected loss of picking each arm."""
    rng = np.random.default_rng(seed)
    pa = rng.beta(1 + x_a, 1 + n_a - x_a, draws)
    pb = rng.beta(1 + x_b, 1 + n_b - x_b, draws)
    return {
        "prob_b_better": float((pb > pa).mean()),
        "expected_loss_choose_b": float(np.maximum(pa - pb, 0).mean()),
        "expected_loss_choose_a": float(np.maximum(pb - pa, 0).mean()),
    }


# --------------------------------------------------------------------------- #
# Power
# --------------------------------------------------------------------------- #
def mde_two_prop(p_base: float, n_per_arm: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest absolute difference detectable with the given sample (normal approximation)."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(z * np.sqrt(2 * p_base * (1 - p_base) / n_per_arm))


def n_per_arm(p_base: float, abs_diff: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Players needed in each arm to detect abs_diff (unpooled normal approximation)."""
    p2 = p_base + abs_diff
    z_a, z_b = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    n = (z_a + z_b) ** 2 * (p_base * (1 - p_base) + p2 * (1 - p2)) / abs_diff ** 2
    return int(np.ceil(n))


def power_two_prop(p_a: float, p_b: float, n_per_arm: float, alpha: float = 0.05) -> float:
    """Power of the two-sided z-test for true rates p_a and p_b."""
    se = np.sqrt(p_a * (1 - p_a) / n_per_arm + p_b * (1 - p_b) / n_per_arm)
    zc = stats.norm.ppf(1 - alpha / 2)
    d = abs(p_b - p_a) / se
    return float(stats.norm.sf(zc - d) + stats.norm.cdf(-zc - d))
