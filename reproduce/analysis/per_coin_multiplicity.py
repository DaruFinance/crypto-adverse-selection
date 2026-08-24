"""
Rebuild the per-coin surface with a multiplicity correction applied.

Run from the repository root: python reproduce/analysis/per_coin_multiplicity.py
--out per_coin_multiplicity.rebuilt.json

The per-coin surface is 96 testable cells, 48 coins across two markout
horizons, and the paper reads it as a description of dispersion rather than as
96 tests. That reading is defensible and it is not self-enforcing: a reader who
counts how many cells clear zero is running 96 tests whatever the surrounding
prose says. This producer runs them properly so the count can be quoted either
way.

Each cell's p-value is recovered from the shipped month-clustered t interval
rather than recomputed from the panel, so it is the same interval the verdict
column already reads from. The standard error is the interval's half-width over
the critical value it was built with, and the degrees of freedom are the ones
the shipped estimator used, the Kish effective month count floored and reduced
by one. A cell that withheld its verdict has no interval worth inverting and is
carried as NaN.

The family is the cells that returned a verdict. Correcting across all 96 would
charge the panel for tests it declined to run, which is the opposite of what the
abstention rule is for, so the withheld cells leave the family rather than
entering it with a missing p-value. Both family definitions are reported, the 70
verdict-returning cells across both horizons and the 10-second cells alone,
because a reader quoting the surface is usually quoting one horizon.

Benjamini-Hochberg is the right correction here rather than a family-wise one.
The question a reader asks of this surface is how much of it holds up, not
whether any single cell survives against the whole family, and the cells are
positively dependent through shared months and shared venue-level shocks, which
is the regime BH is stable in. The Benjamini-Yekutieli variant, valid under
arbitrary dependence, is reported beside it so the cost of the weaker assumption
is visible instead of argued about.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

HERE = Path(__file__).resolve().parent
REPRODUCE = HERE.parent
sys.path.insert(0, str(REPRODUCE.parent))

from makercex import benjamini_hochberg, t_crit_95, t_two_sided_p

SOURCE = REPRODUCE / "per_coin_intervals.json"
HORIZONS = ("10s", "60s")
ALPHA = 0.05


def cell_p(cell):
    """Invert the shipped month-clustered t interval back to a p-value."""
    n_eff = cell.get("n_effective_months_kish")
    interval = cell.get("ci95_t")
    point = cell.get("net_bp")
    if not cell.get("verdict_is_available") or interval is None or point is None:
        return float("nan"), float("nan"), None
    if n_eff is None or not np.isfinite(n_eff):
        return float("nan"), float("nan"), None
    df = max(1, int(np.floor(n_eff))) - 1
    crit = t_crit_95(df)
    se = (interval[1] - interval[0]) / (2.0 * crit)
    if not np.isfinite(se) or se <= 0:
        return float("nan"), float("nan"), df
    t_stat = point / se
    return t_stat, t_two_sided_p(t_stat, df), df


def benjamini_yekutieli(pvalues):
    """BY: BH on p times the harmonic number of the family size."""
    p = np.asarray(pvalues, dtype=np.float64)
    live = np.isfinite(p)
    m = int(live.sum())
    if m == 0:
        return np.full(p.shape, np.nan)
    c_m = float(np.sum(1.0 / np.arange(1, m + 1)))
    scaled = np.where(live, p * c_m, np.nan)
    return np.minimum(benjamini_hochberg(scaled), 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(SOURCE))
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    a = ap.parse_args()

    data = json.loads(Path(a.source).read_text())
    cells = []
    for venue, block in data["venues"].items():
        for coin, per_horizon in sorted(block["per_coin"].items()):
            for tau in HORIZONS:
                cell = per_horizon.get(tau)
                if cell is None or not cell.get("testable"):
                    continue
                t_stat, p, df = cell_p(cell)
                cells.append({
                    "venue": venue, "coin": coin, "horizon": tau,
                    "net_bp": cell.get("net_bp"),
                    "t_stat": t_stat, "p_raw": p,
                    "t_degrees_of_freedom": df,
                    "verdict_is_available": bool(cell.get("verdict_is_available")),
                    "clears_zero_shipped": bool(cell.get("clears_zero")),
                })

    p_all = np.array([c["p_raw"] for c in cells])
    q_all = benjamini_hochberg(p_all)
    qy_all = benjamini_yekutieli(p_all)
    p_10s = np.array([c["p_raw"] if c["horizon"] == "10s" else np.nan for c in cells])
    q_10s = benjamini_hochberg(p_10s)

    for c, q, qy, q1 in zip(cells, q_all, qy_all, q_10s):
        c["q_bh_both_horizons"] = float(q) if np.isfinite(q) else None
        c["q_by_both_horizons"] = float(qy) if np.isfinite(qy) else None
        c["q_bh_10s_only"] = float(q1) if np.isfinite(q1) else None

    live = [c for c in cells if np.isfinite(c["p_raw"])]
    # The shipped verdict column is the same interval read as a yes or no, so a
    # disagreement here would mean the p-value inversion is wrong rather than
    # that the correction changed anything. It is checked, not assumed.
    mismatch = [f"{c['venue']}/{c['coin']}/{c['horizon']}" for c in live
                if (c["p_raw"] < 0.05) != c["clears_zero_shipped"]]

    def count(key, alpha):
        return sum(1 for c in live if c[key] is not None and c[key] < alpha)

    out = {
        "source": Path(a.source).name,
        "alpha": a.alpha,
        "n_testable_cells": len(cells),
        "n_cells_in_family": len(live),
        "n_cells_withheld": len(cells) - len(live),
        "family": ("the cells that returned a verdict; withheld cells leave the "
                   "family rather than entering it with a missing p-value"),
        "counts": {
            "raw_below_alpha": count("p_raw", a.alpha),
            "bh_below_alpha_both_horizons": count("q_bh_both_horizons", a.alpha),
            "by_below_alpha_both_horizons": count("q_by_both_horizons", a.alpha),
            "raw_below_alpha_10s": sum(1 for c in live if c["horizon"] == "10s"
                                       and c["p_raw"] < a.alpha),
            "bh_below_alpha_10s": sum(1 for c in live if c["horizon"] == "10s"
                                      and c["q_bh_10s_only"] is not None
                                      and c["q_bh_10s_only"] < a.alpha),
            "n_cells_10s_in_family": sum(1 for c in live if c["horizon"] == "10s"),
        },
        "cells_lost_to_bh": sorted(
            f"{c['venue']}/{c['coin']}/{c['horizon']}" for c in live
            if c["p_raw"] < a.alpha and (c["q_bh_both_horizons"] or 1.0) >= a.alpha),
        "shipped_verdict_disagreements": mismatch,
        "cells": cells,
        "limitations": {
            "p_values_are_inverted_not_recomputed": (
                "Each p-value comes from inverting the shipped month-clustered t "
                "interval, so it inherits every property of that interval, "
                "including the under-coverage the coverage study reports under "
                "right-skewed coin effects. A correction for multiplicity does "
                "nothing about a marginal test that is the wrong size."),
            "the_cells_are_not_independent": (
                "Cells share months, share venues and share the two horizons of "
                "one coin, so the family is positively dependent rather than "
                "independent. BH is stable under positive dependence and the BY "
                "column prices the arbitrary-dependence case beside it."),
            "two_horizons_of_one_coin_are_one_measurement": (
                "The 10-second and 60-second cells of a coin differ only in the "
                "adverse leg, so the 96-cell family double counts each coin. The "
                "10-second-only family is reported for that reason and is the "
                "one to quote."),
        },
    }
    Path(a.out).write_text(json.dumps(out, indent=2, default=float) + "\n")

    c = out["counts"]
    print(f"{out['n_testable_cells']} testable cells, {out['n_cells_in_family']} in the "
          f"family, {out['n_cells_withheld']} withheld")
    print(f"  both horizons: {c['raw_below_alpha']} clear at raw p<{a.alpha}, "
          f"{c['bh_below_alpha_both_horizons']} survive BH, "
          f"{c['by_below_alpha_both_horizons']} survive BY")
    print(f"  10s only:      {c['raw_below_alpha_10s']} of {c['n_cells_10s_in_family']} "
          f"clear at raw p<{a.alpha}, {c['bh_below_alpha_10s']} survive BH")
    if out["cells_lost_to_bh"]:
        print(f"  lost to BH: {', '.join(out['cells_lost_to_bh'])}")
    if mismatch:
        print(f"  WARNING: {len(mismatch)} cells disagree with the shipped verdict: "
              f"{', '.join(mismatch)}")


if __name__ == "__main__":
    main()
