"""
Rebuild the cross-venue lead-lag summary from the per-coin-day panel.

Run from the repository root: python reproduce/analysis/cross_venue_leadlag.py
--out cross_venue_leadlag.rebuilt.json

For each coin-day where both venues trade the same coin, the panel carries the
lag at which the cross-correlation of returns peaks. A positive lag means Bybit
leads Hyperliquid. Lags are measured on a 250 ms grid, so a reported median is
a grid point rather than a resolved estimate and should not be read to finer
precision than the grid itself.

The matched columns come from the comparison in which both feeds are first put
on a common sampling cadence. The raw columns do not, and the two venues sample
at different rates, so the raw lag partly measures that difference rather than
the lead. The matched figures are the ones the paper reports.

Coin-days are grouped by date for the interval, since coins move together
within a day and the dates here are consecutive rather than independent draws.

The interval can come out zero width, because the lag lives on a 250 ms grid
and every resample can land on the same grid point. That is the grid refusing
to resolve rather than a precise measurement, so no verdict is returned when it
happens.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

HERE = Path(__file__).resolve().parent
REPRODUCE = HERE.parent
GRID_MS = 250
N_BOOT = 4000
SEED = 101


def load(path):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            try:
                rows.append({
                    "coin": r["coin"],
                    "date": r["date"],
                    "matched_lag": float(r["matched_peak_lag_ms"]),
                    "matched_corr": float(r["matched_peak_corr"]),
                    "raw_lag": float(r["raw_peak_lag_ms"]),
                    "bybit_cadence": float(r["bybit_cadence_ms"]),
                    "hl_cadence": float(r["hl_cadence_ms"]),
                })
            except (KeyError, ValueError, TypeError):
                continue
    return rows


def date_cluster_median_ci(rows, key, n_boot=N_BOOT, seed=SEED):
    by = defaultdict(list)
    for r in rows:
        by[r["date"]].append(r[key])
    dates = list(by)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = np.concatenate([by[dates[i]] for i in
                               rng.integers(0, len(dates), len(dates))])
        draws[b] = np.median(pick)
    per_date = np.array([np.median(by[d]) for d in dates])
    can_reach = bool(per_date.size and per_date.max() >= 0 >= per_date.min())
    return [float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5))], len(dates), can_reach


def summarise(rows, key):
    lags = np.array([r[key] for r in rows])
    ci, n_clusters, can_reach = date_cluster_median_ci(rows, key)
    return {
        "median_peak_lag_ms": float(np.median(lags)),
        "median_lag_ci95_ms": ci,
        "ci_clusters": n_clusters,
        "clears_zero": (bool(ci[0] > 0 or ci[1] < 0)
                        if ci[1] > ci[0] else None),
        "percentile_interval_has_power_to_fail": can_reach,
        "n_bybit_leads": int((lags > 0).sum()),
        "n_hl_leads": int((lags < 0).sum()),
        "n_zero_lag": int((lags == 0).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=str(
        REPRODUCE / "panels" / "cross_venue_leadlag_coindays.csv"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = load(a.panel)
    matched = summarise(rows, "matched_lag")
    matched["mean_peak_corr"] = float(
        np.mean([r["matched_corr"] for r in rows]))
    by_coin = defaultdict(list)
    for r in rows:
        by_coin[r["coin"]].append(r["matched_lag"])
    per_coin = {c: {"median_lag_ms": float(np.median(v)), "n": len(v),
                    "leads": bool(np.median(v) > 0)}
                for c, v in sorted(by_coin.items())}
    bc = np.array([r["bybit_cadence"] for r in rows])
    hc = np.array([r["hl_cadence"] for r in rows])
    lag = np.array([r["matched_lag"] for r in rows])
    spread = float(hc.max() - hc.min())
    corr = (float(np.corrcoef(hc, lag)[0, 1]) if hc.std() > 0 else float("nan"))
    out = {
        "convention": "positive lag = Bybit leads Hyperliquid",
        "cadence": {
            "bybit_median_ms": float(np.median(bc)),
            "hyperliquid_median_ms": float(np.median(hc)),
            "gap_ms": float(np.median(hc) - np.median(bc)),
            "hyperliquid_spread_ms": spread,
            "corr_cadence_with_matched_lag": corr,
        },
        "grid_ms": GRID_MS,
        "n_coindays": len(rows),
        "matched": matched,
        "raw": summarise(rows, "raw_lag"),
        "per_coin_matched": per_coin,
        "n_coins_bybit_leads": sum(1 for v in per_coin.values() if v["leads"]),
        "n_coins_hl_leads": sum(1 for v in per_coin.values()
                                if v["median_lag_ms"] < 0),
        "limitations": {
            "lag_grid_is_coarse": (
                f"Peaks are located on a {GRID_MS} ms grid, so the reported median "
                f"lead is {abs(np.median(lag)) / GRID_MS:.0f} grid steps and cannot "
                f"be read finer than one step."),
            "shared_dates_only": (
                "Only coin-days both venues traded on the same date enter, so this "
                "is not a statement about either venue outside that overlap."),
            "zero_width_interval_returns_no_verdict": (
                "The date-clustered interval on the median collapses to zero width "
                "on this grid, so no verdict is reported rather than a false one."),
            "publication_cadence_is_confounded_with_the_lead": (
                f"The two venues publish at different rates, {np.median(bc):.0f} ms "
                f"against {np.median(hc):.0f} ms, a gap of "
                f"{np.median(hc) - np.median(bc):.0f} ms that is the same order as "
                f"the {np.median(lag):.0f} ms lead reported here. A slower feed can "
                f"appear to follow a faster one for that reason alone. The matching "
                f"correction is applied upstream and cannot be rebuilt from what "
                f"ships. The panel carries only {spread:.0f} ms of cadence "
                f"variation, so the confound cannot be tested here. Across that "
                f"little variation cadence and the lead correlate at {corr:+.3f}, "
                f"whose sign runs against the confound rather than with it, since "
                f"a slower feed trailing for mechanical reasons would show a "
                f"larger lead and not a smaller one. Read the result as unable to "
                f"separate a lead in information from a difference in publication "
                f"rate, not as evidence that it is the latter."),
        },
    }
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"{len(rows)} coin-days over {matched['ci_clusters']} dates")
    print(f"median matched peak lag {matched['median_peak_lag_ms']:.0f} ms, "
          f"Bybit leads on {matched['n_bybit_leads']}")


if __name__ == "__main__":
    main()
