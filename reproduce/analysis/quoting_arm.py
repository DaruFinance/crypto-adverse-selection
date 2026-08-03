"""
Rebuild the Avellaneda-Stoikov quoting arm from its per-coin-day panel.

Run from the repository root:

    python reproduce/analysis/quoting_arm.py --out arm.json

The arm quotes with an inventory-skewed reservation price and is scored against
a passive quote at the touch on the same coin-days. Both legs are measured per
fill, so the comparison is weighted by the fills each leg actually took, and
the difference between them uses the smaller of the two fill counts on each
coin-day so that neither leg's volume drives the average on its own.

Two readings of the same arm point opposite ways and both are reported. Per
fill at no rebate the arm looks better than the touch, because it declines most
of the fills the touch takes and the ones it declines are the worst. Per
quoting opportunity at the venue's published tier the preference reverses,
because a rebate is paid per fill and the arm has given up the volume that
earns it. Only the second describes a maker choosing between the two.

Two intervals are given for the difference. Resampling coin-days treats 91 of
them as 91 independent draws when they come from 13 coins, and the shipped
result file carries that one. Clustering on coins gives roughly +0.167 to +0.704 bp,
about half again as wide, but that percentile interval could not have crossed
zero because all 13 coin means share a sign. On the t interval the
coin-clustered bound runs -0.011 to +0.743 and does not clear zero while the
coin-day bound does, so the verdict does turn on the choice of unit.

By its own criterion the tested family is a depth rule rather than a faithful
Avellaneda-Stoikov quoter, since the median quote sits outside the book more
than half the time. The fill threshold also binds on the arm and not on the
touch, so the coin-days it drops are the ones where it quoted deepest.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

HERE = Path(__file__).resolve().parent
REPRODUCE = HERE.parent
sys.path.insert(0, str(REPRODUCE.parent))

from makercex import cluster_bootstrap
N_BOOT = 4000
SEED = 11
REBATES = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)
TIER_BP = 1.5


def load(path):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            try:
                rows.append({
                    "coin": r["coin"],
                    "date": r["date"],
                    "touch_net": float(r["touch_net_bp"]),
                    "touch_fills": float(r["touch_fills"]),
                    "as_net": float(r["as_net_bp"]),
                    "as_fills": float(r["as_fills"]),
                    "diff": float(r["diff_bp"]),
                    "opportunities": float(r["n_opportunities"]),
                    "outside_book": float(r["as_frac_quotes_outside_book"]),
                    "at_deepest": float(r["as_frac_quotes_at_deepest_level"]),
                })
            except (KeyError, ValueError, TypeError):
                continue
    return rows


def weighted(values, weights):
    w = np.asarray(weights, dtype=np.float64)
    return float(np.average(np.asarray(values, dtype=np.float64), weights=w)
                 ) if w.sum() > 0 else float("nan")


def resample_diff(rows, unit, n_boot=N_BOOT, seed=SEED):
    """Cluster bootstrap on the per-fill difference, through the shared library."""
    values = np.array([r["diff"] for r in rows], dtype=np.float64)
    weights = np.array([min(r["as_fills"], r["touch_fills"]) for r in rows],
                       dtype=np.float64)
    ids = [r["coin"] if unit == "coin" else str(i)
           for i, r in enumerate(rows)]
    return cluster_bootstrap(values, weights, ids, n_boot=n_boot, seed=seed)


def frontier(rows):
    opp = sum(r["opportunities"] for r in rows)
    out = {}
    for reb in REBATES:
        touch = sum(r["touch_fills"] * (r["touch_net"] + reb) for r in rows) / opp
        arm = sum(r["as_fills"] * (r["as_net"] + reb) for r in rows) / opp
        out[f"{reb:.4f}"] = {
            "touch_bp_per_opportunity": touch,
            "as_bp_per_opportunity": arm,
            "prefers": "A-S" if arm > touch else "touch",
        }
    return out


def crossing_rebate(rows):
    opp = sum(r["opportunities"] for r in rows)
    tf = sum(r["touch_fills"] for r in rows) / opp
    af = sum(r["as_fills"] for r in rows) / opp
    tm = sum(r["touch_fills"] * r["touch_net"] for r in rows) / opp
    am = sum(r["as_fills"] * r["as_net"] for r in rows) / opp
    den = tf - af
    return float((am - tm) / den) if abs(den) > 1e-12 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=str(
        REPRODUCE / "panels" / "bybit_quoting_arm_coindays.csv"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = load(a.panel)
    as_net = weighted([r["as_net"] for r in rows], [r["as_fills"] for r in rows])
    common = [min(r["as_fills"], r["touch_fills"]) for r in rows]
    diff = weighted([r["diff"] for r in rows], common)
    n_coins = len({r["coin"] for r in rows})
    b_coin = resample_diff(rows, "coin")
    b_day = resample_diff(rows, "coinday")
    ci_coin, ci_day = b_coin["ci95"], b_day["ci95"]
    n_neg = sum(1 for r in rows if r["as_net"] < 0)
    fill_share = (sum(r["as_fills"] for r in rows)
                  / sum(r["touch_fills"] for r in rows))
    outside = float(np.median([r["outside_book"] for r in rows]))
    fr = frontier(rows)
    out = {
        "n_coindays": len(rows),
        "n_coins": n_coins,
        "as_net_bp": as_net,
        "as_net_negative_on": f"{n_neg}/{len(rows)}",
        "as_fill_share_of_touch": fill_share,
        "as_breakeven_rebate_bp": -as_net,
        "diff_bp": diff,
        "diff_ci95_coin_clustered": ci_coin,
        "diff_ci95_coinday": ci_day,
        "diff_ci95_t_coin_clustered": b_coin["ci95_t"],
        "diff_ci95_t_coinday": b_day["ci95_t"],
        "diff_clears_zero_coin_clustered": (
            b_coin["clears_zero"] if b_coin["verdict_is_available"] else None),
        "diff_clears_zero_coinday": (
            b_day["clears_zero"] if b_day["verdict_is_available"] else None),
        "diff_clears_zero_coin_clustered_percentile":
            bool(ci_coin[0] > 0 or ci_coin[1] < 0),
        "diff_clears_zero_coinday_percentile":
            bool(ci_day[0] > 0 or ci_day[1] < 0),
        "diff_verdict_is_available_coin_clustered": b_coin["verdict_is_available"],
        "diff_percentile_interval_has_power_to_fail_coin_clustered":
            b_coin["percentile_interval_can_reach_zero"],
        "crossing_rebate_bp": crossing_rebate(rows),
        "median_quotes_outside_book": outside,
        "at_deepest_median": float(np.median([r["at_deepest"] for r in rows])),
        "tested_family_is_a_depth_rule": bool(outside > 0.5),
        "rebate_frontier": fr,
        "prefers_at_published_tier": (
            "A-S" if (sum(r["as_fills"] * (r["as_net"] + TIER_BP) for r in rows)
                      > sum(r["touch_fills"] * (r["touch_net"] + TIER_BP)
                            for r in rows)) else "touch"),
        "verdict": "the Avellaneda-Stoikov arm does not escape adverse selection",
    }
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"{len(rows)} coin-days over {n_coins} coins")
    print(f"  arm net {as_net:+.4f} bp, negative on {n_neg}/{len(rows)}, "
          f"taking {fill_share:.4f} of the touch's fills")
    print(f"  per-fill difference {diff:+.4f} bp")
    print(f"    coin-clustered [{ci_coin[0]:+.4f}, {ci_coin[1]:+.4f}]")
    print(f"    by coin-day    [{ci_day[0]:+.4f}, {ci_day[1]:+.4f}]")
    print(f"  at the published tier the maker prefers the "
          f"{out['prefers_at_published_tier']}")


if __name__ == "__main__":
    main()
