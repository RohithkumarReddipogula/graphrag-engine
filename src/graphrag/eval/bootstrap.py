"""Percentile bootstrap confidence intervals (10,000 resamples, fixed seed)."""

import numpy as np

N_RESAMPLES = 10_000


def mean_ci(values: list[float], seed: int = 0, level: float = 0.95) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(N_RESAMPLES, len(x)))].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"mean": round(float(x.mean()), 4), "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4), "n": len(x)}


def paired_diff_ci(a: list[float], b: list[float], seed: int = 0, level: float = 0.95) -> dict[str, float]:
    """CI of mean(a - b) over the same questions."""
    assert len(a) == len(b)
    return mean_ci([x - y for x, y in zip(a, b)], seed=seed, level=level)
