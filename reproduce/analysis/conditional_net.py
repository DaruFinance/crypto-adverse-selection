"""
Rebuild the conditional net contrast and its date-clustered sign-flip test.

Run from the repository root: python reproduce/analysis/conditional_net.py
--block asia --out conditional_net_asia.rebuilt.json

The contrast is the high-volatility minus low-volatility net markout, formed
per coin-day so that the shared per-date sign in the null applies to a
difference rather than to a pooled level. Cells enter only where both regimes
carry at least one fill on that coin-day, and each is weighted by its
calm-regime fill count.

Dependence across coins on the same date is the reason for the design. Applying
one shared sign per date carries whatever same-day dependence exists into the
null by construction, so no correlation matrix is estimated, nothing is imputed
and no repair step is needed. Block lengths group consecutive dates, and the
null is enumerated exactly whenever the block count allows it rather than
sampled.

The two conditioning axes, neither of which is defined anywhere else here.

`block` is the UTC hour of the fill placed into one of three eight-hour session
blocks: asia covers 00 to 07, europe covers 08 to 15 and us covers 16 to 23.
Crypto trades around the clock, so these label the tape rather than any
exchange session.

`regime` is a tercile of realised volatility measured over the 30 minutes
ending at each fill, from mids strictly before it, so the measure cannot
consult a price at or after the fill it labels. The cutoffs are expanding
rather than pooled: each date takes its cutoffs from fills on strictly earlier
dates alone. Dates without enough history are labelled -1 and dropped rather
than defaulted into the middle tercile, which is what makes the cutoffs causal and
not only the measure.

Three consequences worth carrying. The dropped prefix is 3.3 percent of the
tape and is calm-heavy rather than stress-heavy, running 49.4 percent calm
against 21.2 percent stress under pooled labels. The stress share is therefore
not pinned at a third and drifts across the sample, so the calm and stress
cells differ in date composition as well as in coin composition. Pooled cutoffs
restricted to these same dates is the control that would separate the label
channel from the sample channel, and it is not run here.

Only the net component is reported. The gross capture leg admits a choice of
weighting and of which cells to include that was not free, so no p-value is
reported for it.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

HERE = Path(__file__).resolve().parent
REPRODUCE = HERE.parent
COLUMN = {"capture": "capture_sum", "adverse": "adverse_sum", "net": "net_sum"}
BLOCK_DAYS = (1, 3, 5, 10)
RECOMMENDED_BLOCK = 5
EXACT_MAX_BLOCKS = 22
BP = 1e4


CONDITIONING = {
    "block_definition": ("UTC hour of the fill in one of three eight-hour session "
                         "blocks: asia 00 to 07, europe 08 to 15, us 16 to 23"),
    "regime_definition": ("tercile of realised volatility over the 30 minutes ending "
                          "at each fill, from mids strictly before it"),
    "tercile_mode": "expanding",
    "tercile_cutoffs_leak": False,
    "cutoff_rule": ("each date takes cutoffs from fills on strictly earlier dates "
                    "alone; dates without enough history are labelled -1 and dropped "
                    "rather than defaulted into the middle tercile"),
    "dropped_prefix_share_of_tape": 0.033,
    "dropped_prefix_composition_under_pooled_labels": {
        "calm": 0.494, "mid": 0.295, "stress": 0.212},
    "caveats": [
        "the stress share is not pinned at a third and drifts across the sample, so "
        "calm and stress cells differ in date composition as well as coin composition",
        "pooled cutoffs restricted to these same dates is the control that would "
        "separate the label channel from the sample channel, and it is not run here"],
}


def build(sidecar, block, component):
    col = COLUMN[component]
    cell = defaultdict(lambda: defaultdict(lambda: np.zeros(2)))
    with open(sidecar) as fh:
        for r in csv.DictReader(fh):
            if r["block"] != block or r["regime"] not in ("calm", "stress"):
                continue
            cell[r["coin"], r["date"]][r["regime"]] += np.array(
                [float(r[col]), float(r["n_fills"])])
    coins = sorted({c for c, _ in cell})
    dates = sorted({d for _, d in cell})
    ci = {c: i for i, c in enumerate(coins)}
    di = {d: i for i, d in enumerate(dates)}
    diff = np.full((len(coins), len(dates)), np.nan)
    weight = np.zeros_like(diff)
    for (c, d), rg in cell.items():
        if "calm" in rg and "stress" in rg and rg["calm"][1] > 0 and rg["stress"][1] > 0:
            diff[ci[c], di[d]] = (rg["stress"][0] / rg["stress"][1]
                                  - rg["calm"][0] / rg["calm"][1]) * BP
            weight[ci[c], di[d]] = rg["calm"][1]
    have = np.isfinite(diff)
    return coins, dates, np.where(have, diff, 0.0), weight * have


def contrast(diff, weight, signs=None):
    den = weight.sum(axis=1)
    den = np.where(den > 0, den, 1e-9)
    weighted = diff * weight
    if signs is None:
        return weighted.sum(axis=1) / den
    return weighted @ signs / den


def flip_test(diff, weight, n_dates, block_len, stat_obs, n_coins, keep,
              n_boot, rng):
    n_blocks = int(np.ceil(n_dates / block_len))
    den = weight.sum(axis=1)
    den = np.where(den > 0, den, 1e-9)
    weighted = diff * weight
    if n_blocks <= EXACT_MAX_BLOCKS:
        hits = 0
        for start in range(0, 2 ** n_blocks, 4096):
            rows = np.arange(start, min(start + 4096, 2 ** n_blocks))
            bits = ((rows[:, None] >> np.arange(n_blocks)) & 1) * 2.0 - 1.0
            signs = np.repeat(bits, block_len, axis=1)[:, :n_dates]
            mu = signs @ weighted.T / den[None, :]
            pos = (mu[:, keep] > 0).sum(axis=1)
            hits += int((np.maximum(pos, n_coins - pos) >= stat_obs).sum())
        return hits / float(2 ** n_blocks), True, int(2 ** n_blocks), n_blocks
    hits = 0
    for _ in range(n_boot):
        bits = rng.choice((-1.0, 1.0), size=n_blocks)
        signs = np.repeat(bits, block_len)[:n_dates]
        c = contrast(diff, weight, signs)
        pos = int((c[keep] > 0).sum())
        if max(pos, n_coins - pos) >= stat_obs:
            hits += 1
    return (1.0 + hits) / (1.0 + n_boot), False, None, n_blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidecar",
                    default=str(REPRODUCE / "panels" / "bybit_conditional_cells.csv"))
    ap.add_argument("--block", default="asia",
                    choices=("asia", "europe", "us"),
                    help="session block; the three shipped results use all of them")
    ap.add_argument("--component", default="net", choices=sorted(COLUMN))
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    coins, dates, diff, weight = build(a.sidecar, a.block, a.component)
    obs = contrast(diff, weight)
    total_weight = weight.sum(axis=1)
    keep = np.isfinite(obs) & (total_weight > 0)
    n_coins = int(keep.sum())
    n_pos = int((obs[keep] > 0).sum())
    stat_obs = max(n_pos, n_coins - n_pos)
    parsed = [dt.datetime.strptime(d, "%Y%m%d") for d in dates]
    gap = float(np.median([(parsed[i + 1] - parsed[i]).days
                           for i in range(len(parsed) - 1)])) if len(parsed) > 1 else 1.0

    rng = np.random.default_rng(a.seed)
    by_block = {}
    for L in BLOCK_DAYS:
        p, exact, patterns, n_blocks = flip_test(
            diff, weight, len(dates), L, stat_obs, n_coins, keep, a.n_boot, rng)
        by_block[str(L)] = {
            "n_blocks": n_blocks,
            "p": p,
            "p_is_exact_enumeration": exact,
            "n_sign_patterns": patterns,
            "clears_5pct": bool(p < 0.05),
            "median_calendar_days_per_block": float(L * gap),
        }
    out = {
        "block": a.block,
        "component": a.component,
        "n_coins": n_coins,
        "n_coins_positive": n_pos,
        "n_dates": len(dates),
        "n_coins_excluded_zero_weight": int((total_weight <= 0).sum()),
        "by_block_length_by_block_index": by_block,
        "recommended_block_length": RECOMMENDED_BLOCK,
        "recommended_p": by_block[str(RECOMMENDED_BLOCK)]["p"],
        "median_calendar_gap_between_kept_dates_days": gap,
        "conditioning": CONDITIONING,
    }
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"{a.block:7s} {n_pos}/{n_coins} coins positive, "
          f"p {out['recommended_p']:.4f} at block {RECOMMENDED_BLOCK}")


if __name__ == "__main__":
    main()
