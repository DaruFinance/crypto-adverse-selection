"""
Measure the true size of the intervals this repository ships.

Coins and dates are crossed so that every coin appears on every date. Indexing
both off one counter aliases them whenever the two counts share a factor, which
can leave each coin on a single date and turn a market-wide shock into a coin
effect that clustering absorbs. The sweep asserts the design is crossed before
it measures anything.

Run from the repository root: python reproduce/analysis/interval_coverage.py
--out coverage_sweep.rebuilt.json

Every figure the README and the inference module quote about how often these
intervals reject a true null comes from this script. It draws panels with no
effect in them, runs each interval and counts how often each one says there is
something there. A nominal 95 percent interval should reject 5 percent of the
time.

Two studies are written. The first sweeps the conditions below with the
one-way interval clustered on the coin and the date crossed in. The second,
under `paper_configuration`, clusters one-way on the calendar month and crosses
in the coin, which is what every shipped verdict does, and it is the one to
quote for the choice between the month-only and the two-way interval. The first
study cannot speak to that choice: a coin effect is the dimension its own
one-way estimator already clusters on, so both estimators absorb it there.

Four conditions are swept in the first study, because each stresses a
different interval. The
first is a shared effect within a cluster, which is the dependence the
clustering exists to absorb: an interval ignoring it is far too narrow, one
clustering correctly is unmoved. Cluster count is the second: percentile
bootstraps under-cover when clusters are few, and adding rows inside a cluster
does not help. Weight concentration is the third: fill counts span orders of
magnitude across coins, and a t interval taking its degrees of freedom from the
raw cluster count over-rejects badly under that skew. A shock shared by every
coin on the same date is the fourth, and a one-way coin-clustered interval
cannot see it at all.

The wild test is only run up to 13 clusters. Above that it enumerates every
sign vector, which is over a million of them at 20 clusters, and the cost of
that inside a replication sweep is far larger than the rest of the script
combined.

The last column is the fraction of panels on which a verdict was available at
all, which is what falls away under weight concentration rather than the size.

Each rate is reported with its Monte Carlo standard error, which at the default
replication count is about a percentage point near a rate of 0.05. Read the
figures to two digits, not three: the ordering between conditions is stable,
the third digit is not.

Runtime is about two minutes at the default replication count, which makes this
the slowest script here by a wide margin. Lower it with --reps for a quicker
check; the numbers get noisier but the ordering holds.

The sweep resamples 400 times where the shipped results use 4000, so the
percentile bounds it grades are noisier than the ones that ship. The effect
falls on the percentile row rather than on the t row, which is analytic.

Weight is drawn per row and independently of value in most conditions, which is
the easy case. The last two draw it per cluster, and the second of those ties
the weight to the size of the cluster effect, so the heavy clusters are also
the atypical ones. That is the shape the real panels carry. One coin can hold a
third of the fills on an atypical capture, which is the harder case.

Cluster effects are Gaussian unless `effect_sigma` is set, which draws them
lognormal and demeaned instead. NOTE: this parameter is the SIGMA of the
lognormal, not a skewness. A lognormal with sigma s has skewness
(e^{s^2}+2)*sqrt(e^{s^2}-1), so sigma 0.44 is skewness 1.49, sigma 0.55 is
skewness 1.99, and sigma 2.0 is skewness 414. Rows are labelled with both. The sign-flip test names cluster-wise
symmetry as its assumption, and without that switch nothing here stressed it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

HERE = Path(__file__).resolve().parent
REPRODUCE = HERE.parent
sys.path.insert(0, str(REPRODUCE.parent))

from makercex import cluster_bootstrap, wild_cluster_p

NOMINAL = 0.05
WILD_MAX_CLUSTERS = 13


def panel(rng, n_clusters, per_cluster, weight_skew=0.0, date_shock=0.0,
          cluster_effect=0.0, n_dates=10, cluster_weights=False,
          weight_value_corr=0.0, effect_sigma=0.0):
    n = n_clusters * per_cluster
    ids = [f"c{i % n_clusters}" for i in range(n)]
    dates = [f"d{(i // n_clusters) % n_dates}" for i in range(n)]
    shock = rng.standard_normal(n_dates) * date_shock
    if effect_sigma:
        effect = (rng.lognormal(0.0, effect_sigma, n_clusters)
                  - np.exp(effect_sigma * effect_sigma / 2.0)) * cluster_effect
    else:
        effect = rng.standard_normal(n_clusters) * cluster_effect
    values = np.array([shock[(i // n_clusters) % n_dates]
                       + effect[i % n_clusters]
                       + rng.standard_normal() for i in range(n)])
    if not weight_skew:
        weights = np.ones(n)
    elif cluster_weights:
        z = rng.standard_normal(n_clusters)
        if weight_value_corr:
            mag = np.abs(effect)
            sd = mag.std()
            if sd > 0:
                rho = weight_value_corr
                z = (rho * ((mag - mag.mean()) / sd)
                     + np.sqrt(max(0.0, 1 - rho * rho)) * z)
        per = np.exp(z * weight_skew)
        weights = np.array([per[i % n_clusters] for i in range(n)])
    else:
        weights = np.exp(rng.standard_normal(n) * weight_skew)
    return values, weights, ids, dates


def paper_panel(rng, n_months, n_coins, dates_per_month, date_shock=0.0,
                coin_effect=0.0):
    """Rows on a month-by-coin grid with calendar dates nested inside months.

    The sweep above clusters one-way on the coin and crosses in the date. This
    paper does the opposite: it clusters one-way on the calendar month and
    crosses in the coin. The two are not the same study. A shock shared by
    every coin on one date is within-cluster dependence that month clustering
    already absorbs, while a coin effect that persists across months is
    dependence month clustering cannot see at all. Neither condition can be
    read off the coin-clustered grid, so both are measured here in the
    configuration the shipped verdicts actually use.

    Returns values, weights, month labels and coin labels, in that order, so
    the month is the one-way cluster and the coin is the second dimension.
    """
    n_dates = n_months * dates_per_month
    shock = rng.standard_normal(n_dates) * date_shock
    effect = rng.standard_normal(n_coins) * coin_effect
    values, months, coins = [], [], []
    for d in range(n_dates):
        for c in range(n_coins):
            values.append(shock[d] + effect[c] + rng.standard_normal())
            months.append(f"m{d // dates_per_month}")
            coins.append(f"c{c}")
    return np.array(values), np.ones(len(values)), months, coins


def paper_size(reps, seed, **kw):
    """Rejection rate of the month-only and month-by-coin intervals."""
    hits = {"month_only": 0, "two_way": 0}
    ran = {"month_only": 0, "two_way": 0}
    for s in range(reps):
        rng = np.random.default_rng(seed + s)
        v, w, months, coins = paper_panel(rng, **kw)
        one = cluster_bootstrap(v, w, months, n_boot=400, seed=s)
        if one["verdict_is_available"]:
            ran["month_only"] += 1
            hits["month_only"] += one["clears_zero"]
        two = cluster_bootstrap(v, w, months, n_boot=400, seed=s,
                                cluster_b=coins)
        if two["verdict_is_available"]:
            ran["two_way"] += 1
            hits["two_way"] += two["clears_zero"]
    out = {}
    for k in hits:
        if not ran[k]:
            out[k] = None
            out[k + "_se"] = None
            continue
        p_hat = hits[k] / ran[k]
        out[k] = p_hat
        out[k + "_se"] = float(np.sqrt(p_hat * (1 - p_hat) / ran[k]))
        out[k + "_verdict_available_frac"] = ran[k] / reps
    return out


def assert_crossed(ids, dates, n_clusters, n_dates):
    """Every cluster must appear on every date, or the design is not crossed."""
    seen = defaultdict(set)
    for c, d in zip(ids, dates):
        seen[c].add(d)
    got = {len(v) for v in seen.values()}
    if len(seen) != n_clusters or got != {n_dates}:
        raise SystemExit(
            f"design is not crossed: {len(seen)} clusters carrying "
            f"{sorted(got)} distinct dates against {n_clusters} and {n_dates}. "
            "A shock shared across a date cannot be told from a cluster effect "
            "on a design like this, and the sweep would understate it.")


def size(reps, seed, **kw):
    hits = {"percentile": 0, "t": 0, "two_way": 0, "wild": 0}
    ran = {"percentile": 0, "t": 0, "two_way": 0, "wild": 0}
    for s in range(reps):
        rng = np.random.default_rng(seed + s)
        v, w, ids, dates = panel(rng, **kw)
        if s == 0:
            assert_crossed(ids, dates, kw["n_clusters"],
                           kw.get("n_dates", 10))
        b = cluster_bootstrap(v, w, ids, n_boot=400, seed=s)
        if b["verdict_is_available"]:
            ran["percentile"] += 1
            ran["t"] += 1
            hits["percentile"] += b["clears_zero_percentile"]
            hits["t"] += b["clears_zero"]
        t = cluster_bootstrap(v, w, ids, n_boot=400, seed=s, cluster_b=dates)
        if t["verdict_is_available"]:
            ran["two_way"] += 1
            hits["two_way"] += t["clears_zero"]
        if kw["n_clusters"] <= WILD_MAX_CLUSTERS:
            p = wild_cluster_p(v, w, ids)
            if np.isfinite(p["p"]):
                ran["wild"] += 1
                hits["wild"] += p["p"] < NOMINAL
    out = {}
    for k in hits:
        if not ran[k]:
            out[k] = None
            out[k + "_se"] = None
            continue
        p_hat = hits[k] / ran[k]
        out[k] = p_hat
        out[k + "_se"] = float(np.sqrt(p_hat * (1 - p_hat) / ran[k]))
    out["verdict_available_frac"] = ran["t"] / reps if reps else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    print("  nominal size is 0.05 throughout\n")
    print(f"  {'condition':34s} {'percentile':>10} {'t':>8} {'two-way':>8} "
          f"{'wild':>8}   {'mc se':>6}   {'avail':>6}")
    for label, kw in (
            ("8 clusters", dict(n_clusters=8, per_cluster=40)),
            ("11 clusters", dict(n_clusters=11, per_cluster=40)),
            ("13 clusters", dict(n_clusters=13, per_cluster=40)),
            ("20 clusters", dict(n_clusters=20, per_cluster=40)),
            ("8 clusters, 300 rows each", dict(n_clusters=8, per_cluster=300)),
            ("13 clusters, weight skew 1.0",
             dict(n_clusters=13, per_cluster=40, weight_skew=1.0)),
            ("13 clusters, weight skew 1.5",
             dict(n_clusters=13, per_cluster=40, weight_skew=1.5)),
            ("13 clusters, weight skew 2.5",
             dict(n_clusters=13, per_cluster=40, weight_skew=2.5)),
            ("13 clusters, weight skew 3.5",
             dict(n_clusters=13, per_cluster=40, weight_skew=3.5)),
            ("13 clusters, coin effect 1.0",
             dict(n_clusters=13, per_cluster=40, cluster_effect=1.0)),
            ("13 clusters, coin effect 2.0",
             dict(n_clusters=13, per_cluster=40, cluster_effect=2.0)),
            ("20 clusters, coin effect 1.0 and skew 1.5",
             dict(n_clusters=20, per_cluster=40, cluster_effect=1.0,
                  weight_skew=1.5)),
            ("13 clusters, weight at cluster level, skew 1.5",
             dict(n_clusters=13, per_cluster=40, weight_skew=1.5,
                  cluster_effect=0.6, cluster_weights=True)),
            ("13 clusters, heavy clusters are the atypical ones",
             dict(n_clusters=13, per_cluster=40, weight_skew=1.5,
                  cluster_effect=0.6, cluster_weights=True,
                  weight_value_corr=0.9)),
            ("13 clusters, coin effects lognormal sigma 0.44 (skewness 1.5)",
             dict(n_clusters=13, per_cluster=40, cluster_effect=0.5,
                  effect_sigma=0.44)),
            ("13 clusters, coin effects lognormal sigma 0.55 (skewness 2.0)",
             dict(n_clusters=13, per_cluster=40, cluster_effect=0.5,
                  effect_sigma=0.55)),
            ("13 clusters, coin effects lognormal sigma 0.8 (skewness 3.7)",
             dict(n_clusters=13, per_cluster=40, cluster_effect=0.5,
                  effect_sigma=0.8)),
            ("13 clusters, coin effects lognormal sigma 1.2 (skewness 11.2)",
             dict(n_clusters=13, per_cluster=40, cluster_effect=0.5,
                  effect_sigma=1.2)),
            ("13 clusters, coin effects lognormal sigma 2.0 (skewness 414)",
             dict(n_clusters=13, per_cluster=40, cluster_effect=0.5,
                  effect_sigma=2.0)),
            ("13 clusters, daily shock 1.0",
             dict(n_clusters=13, per_cluster=40, date_shock=1.0)),
            ("20 clusters, daily shock 1.0",
             dict(n_clusters=20, per_cluster=40, date_shock=1.0)),
    ):
        r = size(a.reps, a.seed, **kw)
        rows.append({"condition": label, **kw, **r})
        wild = "     n/a" if r["wild"] is None else f"{r['wild']:>8.3f}"
        print(f"  {label:34s} {r['percentile']:>10.3f} {r['t']:>8.3f} "
              f"{r['two_way']:>8.3f} {wild}   +/-{r['percentile_se']:.3f}"
              f"   {r['verdict_available_frac']:>6.2f}")

    paper_rows = []
    print(f"\n  the paper's own configuration: one-way on the calendar month, "
          f"crossed with the coin\n")
    print(f"  {'condition':46s} {'month-only':>11} {'two-way':>9}   {'mc se':>6}")
    for label, kw in (
            ("29 months x 13 coins, daily shock inside months",
             dict(n_months=29, n_coins=13, dates_per_month=5, date_shock=1.0)),
            ("29 months x 13 coins, coin effect across months",
             dict(n_months=29, n_coins=13, dates_per_month=5, coin_effect=1.0)),
            ("11 months x 20 coins, daily shock inside months",
             dict(n_months=11, n_coins=20, dates_per_month=5, date_shock=1.0)),
            ("11 months x 20 coins, coin effect across months",
             dict(n_months=11, n_coins=20, dates_per_month=5, coin_effect=1.0)),
    ):
        r = paper_size(a.reps, a.seed, **kw)
        paper_rows.append({"condition": label, **kw, **r})
        print(f"  {label:46s} {r['month_only']:>11.3f} {r['two_way']:>9.3f}"
              f"   +/-{r['month_only_se']:.3f}")

    Path(a.out).write_text(json.dumps(
        {"nominal": NOMINAL, "reps": a.reps, "rows": rows,
         "paper_configuration": {
             "why": ("The rows above cluster one-way on the coin and cross in "
                     "the date. Every shipped verdict clusters one-way on the "
                     "calendar month and crosses in the coin, so the two "
                     "conditions that decide the headline interval choice are "
                     "measured separately here in that configuration."),
             "rows": paper_rows}},
        indent=2, default=float))
    print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
