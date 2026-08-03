"""
Interval and sensitivity around the depth crossing, from the per-level panel.

Run from the repository root: python
reproduce/analysis/depth_rebate_intervals.py --out depth_rebate_intervals.rebuilt.json

The crossing is the maker rebate at which quoting the touch overtakes quoting
the deepest level. It is a ratio of small differences between two levels, so
its sampling error is worth reporting rather than the point alone.

Two resamplings are given. Clustering on coins respects the fact that coin-days
of the same coin are not independent draws, but it narrows the interval here
rather than widening it, because most of the spread across cells is within a
coin rather than between coins. The coin-day resampling is reported next to it
so the narrowing is visible rather than implied.

Neither interval covers the axis that dominates the answer. No maker fee is
charged, on the reading that the rebate replaces it, and charging a residual
fee moves the crossing one for one. The fee curve below shows that directly:
the crossing passes the published tier at a residual smaller than the width of
either interval.
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
N_LEVELS = 5
N_BOOT = 4000
SEED = 11
TIER_BP = 1.5


def load_cells(path, n_levels=N_LEVELS):
    cells = defaultdict(lambda: {"wm": np.zeros(n_levels),
                                 "f": np.zeros(n_levels),
                                 "o": np.zeros(n_levels)})
    with open(path) as fh:
        for r in csv.DictReader(fh):
            i = int(r["level"]) - 1
            if not 0 <= i < n_levels:
                continue
            c = cells[(r["coin"], r["date"])]
            nf = float(r["n_fills"] or 0)
            c["f"][i] += nf
            c["o"][i] += float(r["n_opportunities"] or 0)
            try:
                m = float(r["mean_markout_bp"])
            except (TypeError, ValueError):
                continue
            if np.isfinite(m):
                c["wm"][i] += m * nf
    return dict(cells)


def pooled(cells, keys):
    wm = np.zeros(N_LEVELS)
    f = np.zeros(N_LEVELS)
    o = np.zeros(N_LEVELS)
    for k in keys:
        c = cells[k]
        wm += c["wm"]
        f += c["f"]
        o += c["o"]
    return wm, f, o


def crossing(wm, f, o):
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(f > 0, wm / f, np.nan)
        fr = np.where(o > 0, f / o, np.nan)
    f1, fL = fr[0], fr[-1]
    m1, mL = m[0], m[-1]
    den = f1 - fL
    if not np.isfinite(den) or abs(den) < 1e-12 or abs(den) < 1e-6 * abs(f1):
        return float("nan"), m, fr
    return float((fL * mL - f1 * m1) / den), m, fr


def resample(cells, groups, keys_by_group, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = [groups[i] for i in rng.integers(0, len(groups), len(groups))]
        keys = [k for g in pick for k in keys_by_group[g]]
        x, _, _ = crossing(*pooled(cells, keys))
        if np.isfinite(x):
            draws.append(x)
    return np.array(draws)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel",
                    default=str(REPRODUCE / "panels" / "bybit_depth_levels.csv"))
    ap.add_argument("--tier-bp", type=float, default=TIER_BP)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cells = load_cells(a.panel)
    by_coin = defaultdict(list)
    for k in cells:
        by_coin[k[0]].append(k)
    coins = sorted(by_coin)
    point, m_lvl, fr_lvl = crossing(*pooled(cells, list(cells)))

    coin_draws = resample(cells, coins, by_coin)
    keys_all = list(cells)
    day_draws = resample(cells, keys_all, {k: [k] for k in keys_all})

    loco = {}
    for c in coins:
        keys = [k for cc in coins if cc != c for k in by_coin[cc]]
        loco[c], _, _ = crossing(*pooled(cells, keys))
    live = [v for v in loco.values() if np.isfinite(v)]

    def ci(d):
        return ([float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
                if d.size >= 200 else [float("nan"), float("nan")])

    n_below = int((coin_draws < a.tier_bp).sum()) if coin_draws.size else 0
    at_floor = bool(coin_draws.size and n_below == coin_draws.size)
    out = {
        "venue": "bybit_perp",
        "n_coindays": len(cells),
        "n_coins": len(coins),
        "crossing_bp": point,
        "crossing_definition": "rebate at which the touch overtakes the "
                               "deepest level",
        "markout_bp_by_level": m_lvl.tolist(),
        "fill_rate_by_level": fr_lvl.tolist(),
        "crossing_ci95": ci(coin_draws),
        "coin_bootstrap_ci95": ci(coin_draws),
        "coin_bootstrap_sd": float(coin_draws.std()) if coin_draws.size else float("nan"),
        "coinday_bootstrap_ci95": ci(day_draws),
        "coinday_bootstrap_sd": float(day_draws.std()) if day_draws.size else float("nan"),
        "clustering_unit": "coin",
        "n_clusters": len(coins),
        "leave_one_coin_out_crossing": loco,
        "loco_range": [min(live), max(live)] if live else None,
        "prob_crossing_below_published_tier": (
            n_below / coin_draws.size if coin_draws.size else float("nan")),
        "prob_crossing_below_published_tier_bound": (
            1.0 - 1.0 / (coin_draws.size + 1.0) if at_floor
            else (n_below / coin_draws.size if coin_draws.size else float("nan"))),
        "prob_crossing_below_published_tier_is_at_floor": at_floor,
        "fee_sensitivity": {
            "residual_fee_that_inverts_the_verdict_bp": a.tier_bp - point,
            "curve": {f"{c:.1f}": {"crossing_bp": point + c,
                                   "touch_wins_at_tier": bool(point + c < a.tier_bp)}
                      for c in (0.0, 0.5, 1.0, 1.5, 2.0)},
        },
        "bybit_best_published_tier_bp": a.tier_bp,
    }
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"{len(cells)} coin-days, {len(coins)} coins")
    print(f"crossing {point:.4f} bp")
    print(f"  coin-clustered    {out['coin_bootstrap_ci95']}")
    print(f"  coin-day resample {out['coinday_bootstrap_ci95']}")
    cw = out["crossing_ci95"][1] - out["crossing_ci95"][0]
    dw = out["coinday_bootstrap_ci95"][1] - out["coinday_bootstrap_ci95"][0]
    print(f"  clustering on coins {'narrows' if cw < dw else 'widens'} the "
          f"interval, {cw:.4f} against {dw:.4f} bp")


if __name__ == "__main__":
    main()
