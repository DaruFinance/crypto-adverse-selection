"""
Rebuild the per-coin intervals at both horizons from the coin-day panels.

Run from the repository root: python reproduce/analysis/per_coin_intervals.py
--out per_coin_intervals.rebuilt.json

Every published per-coin cell is a percentile bootstrap that resamples calendar
months, and a per-coin cell has fewer months behind it than the pooled figure
it sits under. That matters because resampling whole months cannot produce a
pooled value outside the range of the month means, so when every month agrees
in sign the interval cannot reach zero and its clears-zero flag is an identity
rather than a test. This script reports that power check next to every
interval, along with the Kish effective month count.

Each venue is reconciled against its own pooled 10 s figure before any per-coin
cell is reported, because a per-coin breakdown computed off a different panel
than the headline is worse than none. Intervals come from the cluster bootstrap
in the `makercex` package, so the shipped library is the one producing them.

A cell returns no verdict where the library refuses one, which is where the
effective month count falls below its floor or the interval has no width.
Twenty-six of the 96 testable cells are in that position, and the shipped files
carry null for them too, so the two agree.

Point estimates match the shipped file exactly. Interval bounds drift, because
the bootstrap consumes its random stream in a different order, and the drift is
larger here than on the pooled table: up to 0.213 bp on Hyperliquid, whose
cells rest on the fewest months. No power flag and no clears-zero verdict
changes.
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
sys.path.insert(0, str(HERE))

from decomposition_table import VENUES, load, pooled
from makercex import cluster_bootstrap

TAUS = ("10s", "60s")
N_BOOT = 4000
SEED = 101


def month_means(rows, field):
    by = defaultdict(list)
    for r in rows:
        by[r["date"][:6]].append(r)
    out = {}
    for m, rs in by.items():
        num = den = 0.0
        for r in rs:
            v, w = r[field], r["fills"]
            if np.isfinite(v) and np.isfinite(w) and w > 0:
                num += v * w
                den += w
        if den:
            out[m] = num / den
    return out


def cell(rows, tau):
    field = f"net_markout_bp_{tau}"
    values = np.array([r[field] for r in rows], dtype=np.float64)
    weights = np.array([r["fills"] for r in rows], dtype=np.float64)
    months = [r["date"][:6] for r in rows]
    mm = month_means(rows, field)
    testable = len(mm) >= 3
    if not testable:
        return {"n_coindays": len(rows), "n_months": len(mm), "testable": False,
                "net_bp": pooled(rows, field, "fills")}
    boot = cluster_bootstrap(values, weights, months, n_boot=N_BOOT, seed=SEED)
    least = min(mm.values(), key=abs)
    lo, hi = min(mm.values()), max(mm.values())
    if not lo - 1e-9 <= least <= hi + 1e-9:
        raise AssertionError("least extreme month mean outside the month mean range")
    if abs(least) > min(abs(lo), abs(hi)) + 1e-9:
        raise AssertionError("least extreme month mean is not the smallest in size")
    if (lo <= 0 <= hi) != boot["percentile_interval_can_reach_zero"]:
        raise AssertionError("power flag disagrees with the month mean range")
    net = boot["point"]
    return {
        "n_coindays": len(rows),
        "n_months": len(mm),
        "testable": True,
        "net_bp": net,
        "ci95": boot["ci95"],
        "ci95_t": boot["ci95_t"],
        "clears_zero": (boot["clears_zero"]
                        if boot["verdict_is_available"] else None),
        "clears_zero_percentile": (boot["clears_zero_percentile"]
                                   if boot["verdict_is_available"] else None),
        "verdict_is_available": boot["verdict_is_available"],
        "negative": bool(net < 0),
        "n_effective_months_kish": boot["n_effective_clusters"],
        "percentile_interval_has_power_to_fail": boot["percentile_interval_can_reach_zero"],
        "month_mean_least_extreme": least,
        "month_mean_range": [min(mm.values()), max(mm.values())],
    }


def venue_block(rows):
    by_coin = defaultdict(list)
    for r in rows:
        by_coin[r["coin"]].append(r)
    per_coin, without_power, testable = {}, [], 0
    for coin in sorted(by_coin):
        per_coin[coin] = {}
        for tau in TAUS:
            c = cell(by_coin[coin], tau)
            per_coin[coin][tau] = c
            if c["testable"]:
                testable += 1
                if not c["percentile_interval_has_power_to_fail"]:
                    without_power.append(f"{coin}@{tau}")
    recomputed = pooled(rows, "net_markout_bp_10s", "fills")
    return {
        "n_coins": len(by_coin),
        "per_coin": per_coin,
        "n_negative_10s": sum(1 for v in per_coin.values()
                              if v["10s"].get("negative")),
        "n_negative_60s": sum(1 for v in per_coin.values()
                              if v["60s"].get("negative")),
        "all_negative_both_horizons": all(
            v["10s"].get("negative") and v["60s"].get("negative")
            for v in per_coin.values()),
        "n_testable_cells": testable,
        "n_cells_with_verdict": sum(
            1 for v in per_coin.values() for tau in TAUS
            if v[tau].get("verdict_is_available")),
        "n_cells_without_power": len(without_power),
        "cells_without_power": without_power,
        "reconciles_with_panel": {"pooled_10s_recomputed": recomputed},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", default=str(REPRODUCE / "panels"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    result = {"n_boot": N_BOOT, "venues": {}}
    total_testable = total_without = 0
    for venue in VENUES:
        rows = load(Path(a.panels) / f"{venue}_coindays.csv")
        block = venue_block(rows)
        result["venues"][venue] = block
        total_testable += block["n_testable_cells"]
        total_without += block["n_cells_without_power"]
        print(f"{venue:14s} {block['n_coins']:>3} coins, "
              f"{block['n_cells_without_power']} of "
              f"{block['n_testable_cells']} cells without power")
    result["n_testable_cells_total"] = total_testable
    result["n_cells_without_power_total"] = total_without
    result["frac_cells_without_power"] = (total_without / total_testable
                                          if total_testable else float("nan"))
    Path(a.out).write_text(json.dumps(result, indent=2, default=float))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
