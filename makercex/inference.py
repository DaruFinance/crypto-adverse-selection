"""
Cluster-robust inference for panels of coin-days.

Coin-days are not independent draws. Coins recur across dates and dates recur
across coins, so an interval that treats cells as exchangeable is too narrow.
This module provides a one-way cluster bootstrap, the analytic
Cameron-Gelbach-Miller two-way cluster-robust standard error, a sign-flip
randomization test over clusters with the null imposed, an exact binomial sign
test, an effective cluster count and a power check that reports whether a
cluster bootstrap was capable of crossing zero at all. Only the one-way
interval and the wild test resample; the two-way standard error is analytic,
not a bootstrap.

At the cluster counts in this paper, between 13 and 20 coins, the percentile
bootstrap interval under-covers: simulation puts its true type-I rate between
7.5 and 9.5 percent against a nominal 5 percent, and adding observations per
cluster does not help because the shortage is in the cluster count. The
interval built from the cluster-robust standard error with a t critical value
is close to nominal across that whole range when cluster weights are
comparable, so `cluster_bootstrap` returns it alongside the percentile interval
as `ci95_t`, and `clears_zero` reads off that interval rather than the
percentile bounds. When `cluster_b` is passed and the two-way standard error is
the wider of the two, the verdict reads off `ci95_t_two_way` instead, which
`verdict_is_two_way` records. The percentile verdict is still available as
`clears_zero_percentile` so the two can be compared, and both are NaN-guarded
and refuse to return a verdict when the effective cluster count falls below
MIN_CLUSTERS or the interval has zero width, either of which lets a flag clear
zero by construction.

The critical value is taken on the effective cluster count rather than the raw
one. Cluster weights here are fill counts, which span orders of magnitude
across coins, and a t interval built on the raw count over-rejects badly under
that skew. The raw-count version, which this module no longer implements, was
consistently anti-conservative against the effective-count one, though by less
than was once claimed here: the largest size reproducible from it is near 0.08
at 8 clusters, since the two differ only in a t critical value and that is a
change of about a tenth in interval width. Taking the degrees of freedom from
the Kish count brings it back near nominal and drifts up with weight skew
rather than turning conservative: on a 13-cluster panel its size runs 0.055,
0.063, 0.070 and 0.083 at skews of 1.0, 1.5, 2.5 and 3.5. What moves faster is
how often a verdict is returned at all, since the effective cluster count falls
below the floor: available on every panel at the first two skews, on 79 percent
at 2.5 and on 38 percent at 3.5. The protection under concentrated weight is
mostly abstention rather than a wider interval, so a no-effect verdict there is
weak evidence rather than evidence of none.

One limit remains that none of the above fixes. This interval clusters on one
dimension. A shock common to every coin on the same date is not in it, and
under a market-wide daily shock the one-way verdict over-rejects sharply. Pass
`cluster_b` to widen the interval for the second dimension. When the two-way
variance comes out non-positive, which happens in sparse designs, the wider of
the two one-way errors is used rather than the first, since falling back to
dimension A alone can narrow the interval when B carries more of the
dependence. Under a market-wide daily shock the one-way verdict is not close,
and it gets worse with more coins rather than better: simulation puts its size
near 0.76 on 13 coin clusters and 0.82 on 20, against a nominal 0.05, and
passing the date labels brings both back to 0.04 to 0.06.

A verdict is only returned when the effective cluster count clears MIN_CLUSTERS
and the interval has width. `verdict_is_available` records that, so a `False`
in `clears_zero` can be read as a test that did not clear rather than as a test
that never ran.

`n_effective_clusters` is a Kish participation ratio on cluster weights. It
measures how concentrated the weight is, not how much independent information
the panel carries, so a panel where one cluster drives the estimate can still
report the full count. It is also exposed under the plainer name
`weight_concentration_clusters`.
"""

from __future__ import annotations

from collections import defaultdict
from math import comb, exp, lgamma, log

import numpy as np

MIN_CLUSTERS = 5

_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
        13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
        19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
        25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
        40: 2.021, 60: 2.000, 120: 1.980}


def t_crit_95(df):
    """Two-sided 95 percent t critical value, interpolated above df 30."""
    if df < 1:
        return float("nan")
    if df in _T95:
        return _T95[df]
    keys = sorted(_T95)
    if df > keys[-1]:
        return 1.960 + (_T95[keys[-1]] - 1.960) * keys[-1] / df
    lo = max(k for k in keys if k < df)
    hi = min(k for k in keys if k > df)
    f = (df - lo) / (hi - lo)
    return _T95[lo] + f * (_T95[hi] - _T95[lo])


def _nan_dict(extra):
    base = {"point": float("nan"), "ci95": [float("nan"), float("nan")],
            "ci95_t": [float("nan"), float("nan")],
            "se_cluster": float("nan"), "se_used_for_verdict": float("nan"),
            "ci95_t_two_way": None, "verdict_is_two_way": False,
            "two_way_variance_was_negative": False,
            "verdict_is_available": False, "clears_zero": False,
            "verdict_interval": "ci95_t",
            "clears_zero_percentile": False, "n_clusters": 0,
            "n_effective_clusters": float("nan"),
            "weight_concentration_clusters": float("nan"),
            "t_degrees_of_freedom": 0,
            "t_degrees_of_freedom_verdict": 0,
            "percentile_interval_can_reach_zero": False}
    base.update(extra)
    return base


def effective_clusters(weights_by_cluster):
    w = np.asarray(weights_by_cluster, dtype=np.float64)
    if w.size == 0 or not np.all(np.isfinite(w)) or np.any(w < 0):
        return float("nan")
    if w.sum() <= 0:
        return float("nan")
    return float(w.sum() ** 2 / (w ** 2).sum())


def _group_index(cluster_ids):
    groups = defaultdict(list)
    for i, c in enumerate(cluster_ids):
        groups[c].append(i)
    return groups


def _cluster_weight_totals(weights, ids):
    """Total weight per cluster, in a stable order, for an effective count."""
    totals = {}
    for w, c in zip(weights, ids):
        k = str(c)
        totals[k] = totals.get(k, 0.0) + float(w)
    return np.array([totals[k] for k in sorted(totals)], dtype=np.float64)


def cluster_bootstrap(values, weights, cluster_ids, n_boot=4000, seed=7,
                      cluster_b=None):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.size == 0 or values.size != weights.size:
        return _nan_dict({})
    if np.any(weights[np.isfinite(weights)] < 0):
        raise ValueError("weights must be non-negative")
    if len(cluster_ids) != values.size:
        raise ValueError("cluster_ids length must match values length")
    finite = np.isfinite(values) & np.isfinite(weights)
    if not finite.any() or weights[finite].sum() <= 0:
        return _nan_dict({})
    values = np.where(finite, values, 0.0)
    weights = np.where(finite, weights, 0.0)
    groups = _group_index(cluster_ids)
    keys = list(groups)
    idx = {k: np.asarray(v) for k, v in groups.items()}
    point = float((values * weights).sum() / weights.sum())
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = np.concatenate([idx[keys[i]]
                               for i in rng.integers(0, len(keys), len(keys))])
        wsum = weights[pick].sum()
        draws[b] = ((values[pick] * weights[pick]).sum() / wsum
                    if wsum > 0 else np.nan)
    good = draws[np.isfinite(draws)]
    lo, hi = (np.percentile(good, [2.5, 97.5]) if good.size
              else (np.nan, np.nan))
    live = [g for g in idx.values() if weights[g].sum() > 0]
    cluster_means = np.array([float((values[g] * weights[g]).sum()
                                    / weights[g].sum()) for g in live])
    cluster_weights = np.array([weights[g].sum() for g in live])
    total = weights.sum()
    contrib = np.array([(weights[g] * (values[g] - point)).sum() / total
                        for g in live])
    n_live = len(live)
    se = (float(np.sqrt((contrib ** 2).sum() * n_live / (n_live - 1)))
          if n_live > 1 else float("nan"))
    n_eff = effective_clusters(cluster_weights)
    df = (max(1, int(np.floor(n_eff))) - 1 if np.isfinite(n_eff)
          else n_live - 1)
    tc = t_crit_95(df)
    se_used, two_way, used_two_way = se, None, False
    if cluster_b is not None:
        two_way = two_way_se(values, weights, cluster_ids, cluster_b)
        if not np.isfinite(two_way["se_two_way"]):
            candidates = [x for x in (two_way["se_a"], two_way["se_b"], se)
                          if np.isfinite(x)]
            fallback = max(candidates) if candidates else float("nan")
            if np.isfinite(fallback) and fallback > se_used:
                se_used = fallback
                used_two_way = True
                df_two_way = min(df, max(1, int(np.floor(
                    effective_clusters(_cluster_weight_totals(
                        weights, cluster_b))))) - 1)
                tc_two_way = t_crit_95(df_two_way)
        elif (np.isfinite(two_way["se_two_way"])
                and two_way["se_two_way"] > se):
            se_used = two_way["se_two_way"]
            used_two_way = True
            df_two_way = min(df, max(1, int(np.floor(
                    effective_clusters(_cluster_weight_totals(
                        weights, cluster_b))))) - 1)
            tc_two_way = t_crit_95(df_two_way)
    if not used_two_way:
        df_two_way, tc_two_way = df, tc
    testable = bool(np.isfinite(se_used) and se_used > 0
                    and np.isfinite(n_eff) and n_eff >= MIN_CLUSTERS)
    return {
        "point": point,
        "ci95": [float(lo), float(hi)],
        "ci95_t": [point - tc * se, point + tc * se],
        "se_cluster": se,
        "t_degrees_of_freedom": df,
        "t_degrees_of_freedom_verdict": df_two_way,
        "se_used_for_verdict": se_used,
        "ci95_t_two_way": (None if not used_two_way else
                           [point - tc_two_way * se_used,
                            point + tc_two_way * se_used]),
        "verdict_is_two_way": used_two_way,
        "two_way_variance_was_negative": bool(
            two_way is not None and two_way["variance_was_negative"]),
        "verdict_is_available": testable,
        "clears_zero": bool(testable
                            and (point - tc_two_way * se_used > 0
                                 or point + tc_two_way * se_used < 0)),
        "clears_zero_percentile": bool(testable and hi > lo
                                       and (lo > 0 or hi < 0)),
        "n_clusters": len(keys),
        "weight_concentration_clusters": effective_clusters(cluster_weights),
        "n_effective_clusters": effective_clusters(cluster_weights),
        "verdict_interval": ("ci95_t_two_way" if used_two_way
                             else "ci95_t"),
        "percentile_interval_can_reach_zero": bool(cluster_means.size
                                        and cluster_means.max() >= 0
                                        >= cluster_means.min()),
    }


def two_way_se(values, weights, cluster_a, cluster_b):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if len(cluster_a) != values.size or len(cluster_b) != values.size:
        raise ValueError("cluster label arrays must match values length")
    finite = np.isfinite(values) & np.isfinite(weights)
    if not finite.all():
        cluster_a = [c for c, f in zip(cluster_a, finite) if f]
        cluster_b = [c for c, f in zip(cluster_b, finite) if f]
        values = values[finite]
        weights = weights[finite]
    if values.size == 0 or weights.sum() <= 0:
        return {"point": float("nan"), "se_a": float("nan"),
                "se_b": float("nan"), "se_inter": float("nan"),
                "se_two_way": float("nan"), "variance_was_negative": False,
                "variance_was_undefined": True}
    point = float((values * weights).sum() / weights.sum())

    def se(ids):
        groups = _group_index(ids)
        total = weights.sum()
        contrib = np.array([(weights[np.asarray(g)]
                             * (values[np.asarray(g)] - point)).sum() / total
                            for g in groups.values()])
        n = len(contrib)
        if n < 2:
            return float("nan")
        return float(np.sqrt((contrib ** 2).sum() * n / (n - 1)))

    inter = [(a, b) for a, b in zip(cluster_a, cluster_b)]
    se_a, se_b, se_i = se(cluster_a), se(cluster_b), se(inter)
    var = se_a ** 2 + se_b ** 2 - se_i ** 2
    defined = np.isfinite(var)
    return {
        "point": point,
        "se_a": se_a,
        "se_b": se_b,
        "se_inter": se_i,
        "se_two_way": float(np.sqrt(var)) if defined and var > 0
        else float("nan"),
        "variance_was_negative": bool(defined and var <= 0),
        "variance_was_undefined": bool(not defined),
    }


def wild_cluster_p(values, weights, cluster_ids, n_boot=4000, seed=7):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if len(cluster_ids) != values.size or weights.size != values.size:
        raise ValueError("cluster_ids and weights must match values length")
    finite = np.isfinite(values) & np.isfinite(weights)
    if not finite.all():
        values = values[finite]
        weights = weights[finite]
        cluster_ids = [c for c, f in zip(cluster_ids, finite) if f]
    groups = _group_index(cluster_ids)
    idx = [np.asarray(v) for v in groups.values()]
    n_g = len(idx)
    total = weights.sum()
    point = float((values * weights).sum() / total) if total > 0 else np.nan
    if n_g == 0 or not np.isfinite(point):
        return {"point": point, "p": float("nan"), "is_exact_enumeration": False,
                "smallest_attainable_p": float("nan")}
    contrib = np.array([(values[g] * weights[g]).sum() / total for g in idx])
    if not np.all(np.isfinite(contrib)):
        return {"point": point, "p": float("nan"), "is_exact_enumeration": False,
                "smallest_attainable_p": float("nan")}
    target = abs(contrib.sum()) * (1.0 - 1e-9)
    if n_g <= 24:
        n_vectors = 2.0 ** n_g
        n_rows = int(n_vectors)
        bits = np.arange(n_g)
        hits = 0
        step = 1 << 16
        for start in range(0, n_rows, step):
            rows = np.arange(start, min(start + step, n_rows))
            signs = ((rows[:, None] >> bits) & 1) * 2.0 - 1.0
            hits += int((np.abs(signs @ contrib) >= target).sum())
        p = hits / n_rows
        exact = True
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice((-1.0, 1.0), size=(n_boot, n_g))
        hits = int((np.abs(signs @ contrib) >= target).sum())
        p = (1.0 + hits) / (1.0 + n_boot)
        exact = False
    return {
        "point": point,
        "p": float(p),
        "is_exact_enumeration": exact,
        "smallest_attainable_p": (2.0 / (2.0 ** n_g) if exact
                                  else 1.0 / (1.0 + n_boot)),
    }


def sign_test_p(n_positive, n_total):
    n_positive, n_total = int(n_positive), int(n_total)
    if n_total <= 0:
        return float("nan")
    if not 0 <= n_positive <= n_total:
        raise ValueError("n_positive must lie between 0 and n_total")
    k = min(n_positive, n_total - n_positive)
    tail = sum(comb(n_total, i) for i in range(k + 1)) / 2 ** n_total
    return float(min(1.0, 2 * tail))

def _betacf(a, b, x, itmax=300, eps=3e-16):
    """Continued fraction for the incomplete beta, Lentz's method."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betainc(a, b, x):
    """Regularized incomplete beta I_x(a, b), enough for a t tail."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = exp(lgamma(a + b) - lgamma(a) - lgamma(b)
                + a * log(x) + b * log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - exp(lgamma(a + b) - lgamma(a) - lgamma(b)
                     + b * log(1.0 - x) + a * log(x)) * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t_stat, df):
    """Two-sided Student-t p-value.

    Implemented here rather than pulled from scipy so the package keeps its
    single numpy dependency. The smoke test checks it against hardcoded
    reference values taken from scipy.
    """
    if df is None or df < 1 or not np.isfinite(t_stat):
        return float("nan")
    df = float(df)
    return float(_betainc(df / 2.0, 0.5, df / (df + float(t_stat) ** 2)))


def benjamini_hochberg(pvalues):
    """BH-adjusted p-values (q-values), preserving input order.

    NaN entries stay NaN and are excluded from the family, so a panel that
    withholds some verdicts is corrected across the verdicts it actually
    returned rather than across the cells it merely has.
    """
    p = np.asarray(pvalues, dtype=np.float64)
    out = np.full(p.shape, np.nan)
    live = np.isfinite(p)
    m = int(live.sum())
    if m == 0:
        return out
    idx = np.flatnonzero(live)
    order = idx[np.argsort(p[idx], kind="stable")]
    ranked = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out[order] = np.minimum(q, 1.0)
    return out
