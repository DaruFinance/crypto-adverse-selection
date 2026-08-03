"""
Rebuild the depth and rebate frontier from the per-level Bybit panel.

Run from the repository root: python
reproduce/analysis/depth_rebate_frontier.py --out
depth_rebate_frontier.rebuilt.json

Level 1 is the touch and level k is the k-th price level measured out from the
mid, so this is quoting distance rather than queue position. For each level the
panel carries a fill-weighted markout, a fill count and an opportunity count
per coin-day, and the frontier aggregates those and then sweeps a maker rebate.

Expected value per quoting opportunity is fill rate times markout plus rebate,
which is the comparison a maker actually faces: quoting deeper improves every
fill and costs most of the fills. The sweep reports the best level at each
rebate, the rebate at which any level first becomes profitable and the rebate
at which the touch overtakes the deepest level.

The measurement covers the bid side of 300 coin-days and charges no maker fee,
on the reading that the rebate replaces it. A residual fee moves the crossing
one for one. The phantom fill share reported in the shipped result file is not
rebuilt here, because no measurement in this repository produces it.

One rebate on the swept grid, 0.8387 bp, is Bybit's pooled net markout rather
than a round number. It comes from the venue decomposition table and is not
derivable from this panel, so it is carried here as a constant.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

HERE = Path(__file__).resolve().parent
REPRODUCE = HERE.parent
N_LEVELS = 5
BEST_TIER_BP = 1.5
BYBIT_BREAKEVEN_BP = 0.8387
REBATES = [0.0, 0.25, 0.5, BYBIT_BREAKEVEN_BP, 1.0, 1.25, BEST_TIER_BP, 2.0]


def collect(path, n_levels=N_LEVELS):
    markout = np.zeros(n_levels)
    weight = np.zeros(n_levels)
    fills = np.zeros(n_levels)
    opportunities = np.zeros(n_levels)
    days = set()
    with open(path) as fh:
        for r in csv.DictReader(fh):
            i = int(r["level"]) - 1
            if not 0 <= i < n_levels:
                continue
            days.add((r["coin"], r["date"]))
            nf = float(r["n_fills"] or 0)
            fills[i] += nf
            opportunities[i] += float(r["n_opportunities"] or 0)
            try:
                m = float(r["mean_markout_bp"])
            except (TypeError, ValueError):
                continue
            if np.isfinite(m) and nf:
                markout[i] += m * nf
                weight[i] += nf
    return (markout / np.where(weight > 0, weight, np.nan),
            fills / np.where(opportunities > 0, opportunities, np.nan),
            fills, opportunities, len(days))


def sweep(markout, fill_rate):
    rows = {}
    for reb in REBATES:
        ev = fill_rate * (markout + reb)
        best = int(np.nanargmax(ev)) + 1
        profitable = bool(np.nanmax(ev) > 0)
        rows[f"{reb:.3f}"] = {
            "ev_by_level": ev.tolist(),
            "best_level": best if profitable else None,
            "any_level_profitable": profitable,
            "argmax_level_even_if_unprofitable": best,
        }
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel",
                    default=str(REPRODUCE / "panels" / "bybit_depth_levels.csv"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    markout, fill_rate, fills, opportunities, n_days = collect(a.panel)
    r1, r5 = fill_rate[0], fill_rate[-1]
    m1, m5 = markout[0], markout[-1]
    crossing = ((r5 * m5 - r1 * m1) / (r1 - r5) if r1 != r5 else float("nan"))
    out = {
        "venue": "bybit_perp",
        "n_coindays": n_days,
        "side": "bid only",
        "level_meaning": "1 = touch (best bid), k = k-th price level from mid",
        "markout_horizon": "10s",
        "markout_weighting": "mean over fill rows, not size-weighted",
        "fee_treatment": "no maker fee charged; rebate replaces it",
        "markout_bp_by_level": markout.tolist(),
        "fill_rate_by_level": fill_rate.tolist(),
        "n_fills_by_level": fills.tolist(),
        "n_opportunities_by_level": opportunities.tolist(),
        "bp_per_opportunity_by_level": (markout * fill_rate).tolist(),
        "fill_rate_ratio_L5_over_L1": float(r5 / r1),
        "by_rebate": sweep(markout, fill_rate),
        "touch_overtakes_deepest_at_rebate_bp": crossing,
        "breakeven_rebate_by_level_bp": (-markout).tolist(),
        "min_rebate_for_any_level_profitable_bp": float(np.nanmin(-markout)),
        "bybit_breakeven_rebate_bp": BYBIT_BREAKEVEN_BP,
        "bybit_best_published_tier_bp": BEST_TIER_BP,
    }
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"{n_days} coin-days, bid side, level 1 is the touch")
    print(f"{'level':>5} {'markout bp':>11} {'fill rate':>11} "
          f"{'bp/opportunity':>15}")
    for i in range(N_LEVELS):
        print(f"{i + 1:>5} {markout[i]:11.4f} {fill_rate[i]:11.6f} "
              f"{markout[i] * fill_rate[i]:15.6f}")
    print(f"touch overtakes the deepest level at {crossing:.4f} bp")


if __name__ == "__main__":
    main()
