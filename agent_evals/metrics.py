"""Metrics that matter: capability (pass@k) vs reliability (pass^k).

pass@k: at least one of k tries succeeds. pass^k: all k succeed.
At 90% per trial, pass@10 is ~100% but pass^10 is only ~35%.
Customer-facing agents need pass^k.
"""

from __future__ import annotations

import math
from math import comb


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from n trials with c successes (Chen et al., 2021)."""
    if not 0 <= c <= n or not 0 < k <= n:
        raise ValueError("need 0 <= c <= n and 0 < k <= n")
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """Unbiased pass^k (all k succeed), the estimator used by tau-bench."""
    if not 0 <= c <= n or not 0 < k <= n:
        raise ValueError("need 0 <= c <= n and 0 < k <= n")
    return comb(c, k) / comb(n, k) if c >= k else 0.0


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a pass rate. On 50 tasks it is about +/-10
    points, so a 2-point change between versions is usually noise."""
    if total == 0:
        return (0.0, 1.0)
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def flips(old: dict[str, bool], new: dict[str, bool]) -> dict[str, list[str]]:
    """Tasks whose outcome changed between two versions. Read these traces:
    an unchanged total can hide fixes and regressions cancelling out."""
    common = old.keys() & new.keys()
    return {
        "fixed": sorted(t for t in common if not old[t] and new[t]),
        "broken": sorted(t for t in common if old[t] and not new[t]),
    }
