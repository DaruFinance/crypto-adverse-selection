"""
Check that each published aggregate uses the weighting its label implies.

Run from the repository root: python reproduce/analysis/weighting_provenance.py
--out checks.json

A weighted mean does not carry its weights with it. Two figures computed from
the same panel under different weightings look equally plausible side by side,
and a label saying which one was used is not evidence that it was. This script
tests each published aggregate against every weighting that could plausibly
have produced it and reports which one actually matches.

The failure it guards against is a diagnostic computed under one weighting
while the statistic it qualifies uses another, or an interval resampling one
unit while its label names a different one. Neither is visible without
recomputing under the alternatives.

Interval clustering is checked the same way, using width rather than the
bounds. A bootstrap seed jitters where an interval sits but not how wide it is,
so width is what identifies the unit that was resampled. That matters because
two of the mislabellings found here were clustering units rather than
weightings.

A pass here means the shipped figure matches its stated weighting and no other,
so the label is doing real work rather than being decoration.
"""

from __future__ import annotations

import argparse
import csv
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

from makercex import cluster_bootstrap

TOL = 1e-9
WIDTH_TOL = 0.02
N_BOOT = 1500
SEED = 101


def load(name):
    return json.loads((REPRODUCE / name).read_text())


def panel(name):
    with open(REPRODUCE / "panels" / name) as fh:
        return list(csv.DictReader(fh))


def wmean(rows, value, weight):
    num = den = 0.0
    for r in rows:
        try:
            v, w = value(r), weight(r)
        except (TypeError, ValueError):
            continue
        if w > 0 and np.isfinite(v):
            num += v * w
            den += w
    return num / den if den else float("nan")


def check(label, stated, candidates):
    """Report which candidate weighting reproduces the shipped figure."""
    errs = {k: max(abs(f() - s) for f, s in pairs) if pairs else float("nan")
            for k, pairs in candidates.items()}
    best = min(errs, key=lambda k: errs[k])
    ok = errs[stated] < TOL and best == stated
    others = ", ".join(f"{k} {errs[k]:.3f}" for k in errs if k != stated)
    print(f"  {'ok  ' if ok else 'FAIL'} {label:38s} stated '{stated}' "
          f"error {errs[stated]:.1e}   alternatives: {others}")
    rivals = sorted(v for k, v in errs.items() if k != stated)
    runner_up = rivals[0] if rivals else float("nan")
    return {"claim": label, "stated_weighting": stated,
            "matches_stated": bool(ok),
            "margin_to_runner_up": float(runner_up - errs[stated]),
            "runner_up_also_within_tolerance": bool(runner_up < TOL),
            "error_by_weighting": {k: float(v) for k, v in errs.items()}}


def check_clustering(label, stated, values, weights, units, shipped_ci):
    """Identify which clustering unit reproduces a shipped interval's width."""
    target = shipped_ci[1] - shipped_ci[0]
    errs = {}
    for name, ids in units.items():
        b = cluster_bootstrap(values, weights, ids, n_boot=N_BOOT, seed=SEED)
        errs[name] = abs((b["ci95"][1] - b["ci95"][0]) - target)
    best = min(errs, key=lambda k: errs[k])
    ok = best == stated and errs[stated] < WIDTH_TOL
    others = ", ".join(f"{k} {errs[k]:.3f}" for k in errs if k != stated)
    print(f"  {'ok  ' if ok else 'FAIL'} {label:38s} stated '{stated}' "
          f"width error {errs[stated]:.3f}   alternatives: {others}")
    rivals = sorted(v for k, v in errs.items() if k != stated)
    runner_up = rivals[0] if rivals else float("nan")
    return {"claim": label, "stated_clustering": stated,
            "matches_stated": bool(ok),
            "margin_to_runner_up": float(runner_up - errs[stated]),
            "runner_up_also_within_tolerance": bool(runner_up < WIDTH_TOL),
            "width_error_by_unit": {k: float(v) for k, v in errs.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    results = []
    print("  each published aggregate against the weightings that could explain it\n")

    dec = load("decomposition_by_venue.json")["venues"]
    for venue in dec:
        rows = panel(f"{venue}_coindays.csv")
        by = defaultdict(list)
        for r in rows:
            by[r["coin"]].append(r)
        shipped = dec[venue]["per_coin"]
        cands = {}
        for scheme, wf in (
                ("fills", lambda r: float(r["n_fills"])),
                ("unweighted", lambda r: 1.0),
                ("coin_units", lambda r: float(r["n_fills"])
                 * float(r["mean_fill_size"] or 0))):
            cands[scheme] = [
                ((lambda rs=rs, wf=wf: wmean(
                    rs, lambda r: float(r["net_markout_bp_10s"]), wf)),
                 shipped[c]["net_bp"])
                for c, rs in by.items() if c in shipped]
        results.append(check(f"{venue} per-coin net", "fills", cands))

    dep = load("depth_rebate_frontier.json")
    rows = panel("bybit_depth_levels.csv")
    by_lvl = defaultdict(list)
    for r in rows:
        by_lvl[int(r["level"])].append(r)
    cands = {}
    for scheme, wf in (("fills", lambda r: float(r["n_fills"] or 0)),
                       ("unweighted", lambda r: 1.0),
                       ("opportunities",
                        lambda r: float(r["n_opportunities"] or 0))):
        cands[scheme] = [
            ((lambda rs=rs, wf=wf: wmean(
                rs, lambda r: float(r["mean_markout_bp"]), wf)),
             dep["markout_bp_by_level"][lv - 1])
            for lv, rs in sorted(by_lvl.items())]
    results.append(check("depth markout by level", "fills", cands))

    ll = load("cross_venue_leadlag.json")["matched"]
    rows = panel("cross_venue_leadlag_coindays.csv")
    matched = np.array([float(r["matched_peak_lag_ms"]) for r in rows])
    raw = np.array([float(r["raw_peak_lag_ms"]) for r in rows])
    cands = {
        "matched_median": [(lambda: float(np.median(matched)),
                            ll["median_peak_lag_ms"])],
        "raw_median": [(lambda: float(np.median(raw)),
                        ll["median_peak_lag_ms"])],
        "matched_mean": [(lambda: float(matched.mean()),
                          ll["median_peak_lag_ms"])],
    }
    results.append(check("lead-lag headline lag", "matched_median", cands))

    print()
    for venue in dec:
        rows = panel(f"{venue}_coindays.csv")
        vals = np.array([float(r["net_markout_bp_10s"]) for r in rows])
        wts = np.array([float(r["n_fills"]) for r in rows])
        units = {"month": [r["date"][:6] for r in rows],
                 "coin": [r["coin"] for r in rows],
                 "date": [r["date"] for r in rows],
                 "coin_day": [f"{r['coin']}|{r['date']}" for r in rows]}
        results.append(check_clustering(
            f"{venue} pooled interval", "month", vals, wts, units,
            dec[venue]["pooled"]["10s_row_mean"]["net_ci95"]))

    n_fail = sum(1 for r in results if not r["matches_stated"])
    print(f"\n  {len(results) - n_fail} of {len(results)} match the weighting "
          f"or clustering their label states")
    Path(a.out).write_text(json.dumps(
        {"n_checks": len(results), "n_failing": n_fail, "checks": results},
        indent=2, default=float))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
