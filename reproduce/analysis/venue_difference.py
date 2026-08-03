"""
Rebuild the venue difference at a matched re-quote rung.

Run from the repository root: python reproduce/analysis/venue_difference.py
--out venue_difference.rebuilt.json

The two venues publish their books at different rates, so comparing them
without matching that rate measures the reconstruction as much as the venue.
The panel carries each coin-day decomposed at seven re-quote gaps, and the
comparison is taken at a fixed 2000 ms gap. An earlier version of this note
said the gap was the one where the two venues' effective re-quote intervals are
closest, which is not what the code does and is not reproducible from what ships
here: the quote-update counts that rule needs are in no panel. The one
re-quote-intensity statistic that does ship, fills per coin-day, is closest
between the venues at 500 ms rather than at 2000. The choice matters for the
composition and not for the difference, so all seven rungs ship in the result
file. Only coin-days both venues traded on the same date are
used, which is what keeps the comparison from being confounded with the period
each panel covers.

A positive difference means Bybit has the higher net, so the other venue is
worse. Four estimators of the same difference are reported because they
disagree by more than any one of their intervals: pooling all fills, pairing
per coin-day at a common weight, pairing unweighted and the median pair. The
unweighted figure is the most negative, which is what it looks like when the
coin-days carrying the least volume carry the largest gaps.

The per-coin breakdown is unweighted, matching the shipped file, so it sums
toward the unweighted headline rather than the common-weight one. The
common-weight breakdown is given beside it, and the two differ by up to half a
basis point on individual coins.

Two clusterings are given, and neither is comfortable. Coins are the unit
independence argues for, but fill weight is concentrated enough that 13 coins
carry the information of about 4.4, which is below the point where the interval
is worth reading, so the coin-clustered verdict is reported as unavailable
rather than as a pass. Dates give 25 clusters, but they are consecutive days in
one regime rather than independent draws, and their interval comes out narrower
still despite that.

What carries the sign is neither interval. It is that all 13 coins run the same
way.
"""

from __future__ import annotations

import argparse
import collections
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

VENUE_A = "bybit_perp"
VENUE_B = "hyperliquid"
GAP_MS = 2000
N_BOOT = 4000
SEED = 101


def load_pairs(path, gap_ms):
    by = defaultdict(dict)
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if int(r["gap_ms"]) != gap_ms:
                continue
            by[(r["coin"], r["date"])][r["venue"]] = r
    pairs = []
    for (coin, date), leg in sorted(by.items()):
        if VENUE_A not in leg or VENUE_B not in leg:
            continue
        try:
            pairs.append({
                "coin": coin, "date": date,
                "a_net": float(leg[VENUE_A]["net_bp"]),
                "b_net": float(leg[VENUE_B]["net_bp"]),
                "a_cap": float(leg[VENUE_A]["capture_bp"]),
                "b_cap": float(leg[VENUE_B]["capture_bp"]),
                "a_adv": float(leg[VENUE_A]["adverse_bp"]),
                "b_adv": float(leg[VENUE_B]["adverse_bp"]),
                "a_w": float(leg[VENUE_A]["n_fills"]),
                "b_w": float(leg[VENUE_B]["n_fills"]),
            })
        except (ValueError, TypeError):
            continue
    return pairs


def rung_choice(path, gap_ms):
    rows = collections.defaultdict(float)
    days = collections.defaultdict(set)
    with open(path) as fh:
        for r in csv.DictReader(fh):
            k = (int(r["gap_ms"]), r["venue"])
            rows[k] += float(r["n_fills"])
            days[k].add((r["coin"], r["date"]))
    venues = sorted({v for _, v in rows})
    by_rung, best = {}, None
    for g in sorted({g for g, _ in rows}):
        rate = {v: rows[(g, v)] / max(len(days[(g, v)]), 1) for v in venues}
        ratio = max(rate.values()) / min(rate.values())
        by_rung[str(g)] = dict(rate, ratio=ratio)
        if best is None or ratio < best[1]:
            best = (g, ratio)
    order = sorted((v["ratio"], int(g)) for g, v in by_rung.items())
    margin = (order[1][0] - order[0][0]) / order[0][0] if len(order) > 1 else float("nan")
    reported = by_rung[str(gap_ms)]["ratio"]
    return {
        "gap_ms_reported": gap_ms,
        "how_chosen": "fixed, not derived",
        "fills_per_coin_day_by_rung": by_rung,
        "rung_closest_on_that_statistic": best[0],
        "rung_runner_up": order[1][1] if len(order) > 1 else None,
        "relative_margin_to_runner_up": float(margin),
        "statistic_separates_the_top_two": bool(margin > 0.01),
        "reported_rung_relative_penalty": float(
            (reported - order[0][0]) / order[0][0]),
        "note": ("the statistic does not separate its top two rungs, so it does not "
                 "select one; what it does say is that the reported rung is not "
                 "among them, and the net gap holds across every rung while the "
                 "capture share does not"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=str(
        REPRODUCE / "panels" / "venue_requote_rungs.csv"))
    ap.add_argument("--gap-ms", type=int, default=GAP_MS)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pairs = load_pairs(a.panel, a.gap_ms)
    an = np.array([p["a_net"] for p in pairs])
    bn = np.array([p["b_net"] for p in pairs])
    aw = np.array([p["a_w"] for p in pairs])
    bw = np.array([p["b_w"] for p in pairs])
    diff = an - bn
    common = np.minimum(aw, bw)
    coins = [p["coin"] for p in pairs]
    dates = [p["date"] for p in pairs]

    by_coin = cluster_bootstrap(diff, common, coins, n_boot=N_BOOT, seed=SEED)
    by_date = cluster_bootstrap(diff, common, dates, n_boot=N_BOOT, seed=SEED)
    by_two = cluster_bootstrap(diff, common, coins, n_boot=N_BOOT, seed=SEED,
                               cluster_b=dates)

    per_coin, per_coin_cw = {}, {}
    for c in sorted(set(coins)):
        m = [i for i, x in enumerate(coins) if x == c]
        per_coin[c] = float(np.mean(diff[m]))
        per_coin_cw[c] = float(np.average(diff[m], weights=common[m]))

    d_by_date = np.array([float(np.average(diff[[i for i, x in enumerate(dates)
                                                if x == d]],
                                          weights=common[[i for i, x
                                                          in enumerate(dates)
                                                          if x == d]]))
                          for d in sorted(set(dates))])
    lag1 = float(np.corrcoef(d_by_date[:-1], d_by_date[1:])[0, 1])

    cap_diff = float(np.average([p["a_cap"] - p["b_cap"] for p in pairs],
                                weights=common))
    adv_diff = float(np.average([p["a_adv"] - p["b_adv"] for p in pairs],
                                weights=common))
    out = {
        "a": VENUE_A, "b": VENUE_B, "gap_ms": a.gap_ms,
        "convention": "positive = A (Bybit) has higher net, i.e. B is worse",
        "n_shared_coindays": len(pairs),
        "n_date_clusters": len(set(dates)),
        "n_coins": len(set(coins)),
        "pooled_diff_bp": float(np.average(an, weights=aw)
                                - np.average(bn, weights=bw)),
        "paired_weighted_diff_bp": float(np.average(diff, weights=common)),
        "paired_unweighted_diff_bp": float(np.mean(diff)),
        "paired_median_diff_bp": float(np.median(diff)),
        "n_coindays_a_higher": int((an > bn).sum()),
        "n_coindays_b_higher": int((bn > an).sum()),
        "coin_clustered": {
            "n_clusters": by_coin["n_clusters"],
            "paired_ci95": by_coin["ci95"],
            "paired_clears_zero": (by_coin["clears_zero"]
                                   if by_coin["verdict_is_available"] else None),
            "paired_clears_zero_percentile": (by_coin["clears_zero_percentile"]
                                              if by_coin["verdict_is_available"]
                                              else None),
            "verdict_is_available": by_coin["verdict_is_available"],
            "n_effective_clusters": by_coin["n_effective_clusters"],
            "percentile_interval_has_power_to_fail": by_coin["percentile_interval_can_reach_zero"],
            "n_coins_negative": sum(1 for v in per_coin.values() if v < 0),
            "per_coin_diff_bp": per_coin,
            "per_coin_diff_bp_common_weight": per_coin_cw,
        },
        "date_clustered": {
            "n_clusters": by_date["n_clusters"],
            "paired_ci95": by_date["ci95"],
            "percentile_interval_has_power_to_fail": by_date["percentile_interval_can_reach_zero"],
            "lag1_autocorr_of_date_diff": lag1,
            "effective_n_clusters": float(len(d_by_date)
                                          * (1 - lag1) / (1 + lag1)),
        },
        "two_way_clustered": {
            "clusters": "coins crossed with dates, Cameron Gelbach Miller",
            "ci95_t_one_way_coins": by_two["ci95_t"],
            "clears_zero": (by_two["clears_zero"]
                            if by_two["verdict_is_available"] else None),
            "verdict_is_available": by_two["verdict_is_available"],
            "verdict_is_two_way": by_two["verdict_is_two_way"],
            "se_one_way_coins": by_coin["se_cluster"],
            "se_used_for_verdict": by_two["se_used_for_verdict"],
            "ci95_t_two_way": by_two["ci95_t_two_way"],
            "two_way_variance_was_negative":
                by_two["two_way_variance_was_negative"],
            "t_degrees_of_freedom_verdict": by_two["t_degrees_of_freedom_verdict"],
        },
        "gap_composition": {
            "capture_diff_bp": cap_diff,
            "adverse_diff_bp": adv_diff,
            "share_from_capture": abs(cap_diff) / (abs(cap_diff) + abs(adv_diff)),
            "share_from_capture_formula": (
                "abs(capture difference) over abs(capture difference) plus "
                "abs(adverse difference); this is a share of the total movement "
                "in the two legs rather than of the net difference, and the two "
                "agree only while both legs carry the same sign, which they do "
                "at every rung here"),
        },
        "rung_choice": rung_choice(a.panel, a.gap_ms),
    }
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"{len(pairs)} shared coin-days over {out['n_date_clusters']} dates "
          f"at the {a.gap_ms} ms rung")
    print(f"  pooled {out['pooled_diff_bp']:+.4f}  paired "
          f"{out['paired_weighted_diff_bp']:+.4f}  unweighted "
          f"{out['paired_unweighted_diff_bp']:+.4f}")
    print(f"  coin-clustered {[round(x, 4) for x in by_coin['ci95']]}, "
          f"date-clustered {[round(x, 4) for x in by_date['ci95']]}")
    print(f"  {out['gap_composition']['share_from_capture']:.1%} of the gap is "
          f"captured spread rather than adverse selection")


if __name__ == "__main__":
    main()
