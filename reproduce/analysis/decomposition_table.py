"""
Rebuild the per-venue decomposition table from the shipped coin-day panels.

Run from the repository root: python reproduce/analysis/decomposition_table.py
--out decomposition_by_venue.rebuilt.json

Reads reproduce/panels/<venue>_coindays.csv and writes the pooled and per-coin
decomposition, using the cluster bootstrap from the `makercex` package. Four of
the eight rebuild scripts route their intervals through it. Three others
bootstrap a median, a ratio of level differences or a sign-flip statistic, none
of which the library covers, so they carry their own loops and the depth
frontier reports no interval at all.

This is not a drop-in replacement for the shipped
reproduce/decomposition_by_venue.json and it takes an explicit --out rather
than overwriting it. Several fields the shipped file carries are not rebuilt
here, among them the fee columns and most of the leave-one-out block, so
copying the output over the shipped file will break the readers that expect
them. Pooled point estimates and pooled interval bounds are identical. Per-coin
interval bounds drift by up to 0.033 bp on Hyperliquid and about 0.007 bp on
the other two venues, because the bootstrap consumes its random stream in a
different order. That is enough to move one knife-edge flag, since Hyperliquid
HYPE sits within 0.0022 bp of zero. It runs in a few seconds.

Intervals resample whole calendar months. Coin-days inside a month share both a
coin and a regime, so an interval treating them as independent draws would be
far too narrow. Two weightings are reported side by side: a mean over fills,
and a weighting by fill count times mean fill size. Order size is a fixed one
coin, so the second is a true size weighting within a coin and is not a mean
over traded value across coins.

A per-coin verdict is withheld where the library withholds one, which is where
the effective month count falls below its floor. The same guard is applied in
the per-coin interval rebuild, and applying it in only one of the two files
left them issuing different flags for the same cell.

The leave-one-out block reports two different things under names that are easy
to confuse. `coins_opposite_sign` is the coins whose own net runs against the
pool, which is what the shipped file and the headline printer mean by it.
`coins_whose_removal_flips_pooled_sign` is the leave-one-out question, and on
these panels it is empty everywhere.

Spread capture is horizon invariant by construction, since capture equals
markout minus adverse drift, which reduces to side times (mid at fill minus
fill price) over mid at fill. Only the adverse term moves with the horizon, so
any claim that the result is horizon robust is a claim about adverse selection
alone.
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
VENUES = ("bybit_perp", "binance_um", "hyperliquid")
TAUS = ("10s", "60s")
FIELDS = ("spread_capture", "adverse_select", "net_markout")
N_BOOT = 4000
SEED = 101


def load(path):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            try:
                fills = float(r["n_fills"])
            except (KeyError, ValueError, TypeError):
                continue
            if fills <= 0:
                continue
            d = {"coin": r["coin"], "date": r["date"], "fills": fills}
            ok = True
            for tau in TAUS:
                for base in FIELDS:
                    for suf in ("", "_szw"):
                        key = f"{base}{suf}_bp_{tau}"
                        try:
                            d[key] = float(r[key])
                        except (KeyError, ValueError, TypeError):
                            d[key] = float("nan")
                            if not suf:
                                ok = False
            try:
                d["size"] = fills * float(r["mean_fill_size"])
            except (KeyError, ValueError, TypeError):
                d["size"] = float("nan")
            if ok:
                rows.append(d)
    return rows


def pooled(rows, field, weight):
    num = den = 0.0
    for r in rows:
        v, w = r[field], r[weight]
        if np.isfinite(v) and np.isfinite(w) and w > 0:
            num += v * w
            den += w
    return num / den if den else float("nan")


def month_bootstrap(rows, field, weight, n_boot=N_BOOT, seed=SEED):
    """Resample calendar months through the shared cluster bootstrap."""
    if len({r["date"][:6] for r in rows}) < 3:
        return None
    values = np.array([r[field] for r in rows], dtype=np.float64)
    weights = np.array([r[weight] for r in rows], dtype=np.float64)
    months = [r["date"][:6] for r in rows]
    return cluster_bootstrap(values, weights, months, n_boot=n_boot, seed=seed)


def month_coin_two_way(rows, field, weight, n_boot=N_BOOT, seed=SEED):
    """The same interval crossed with the coin dimension the months are nested in."""
    if len({r["date"][:6] for r in rows}) < 3 or len({r["coin"] for r in rows}) < 3:
        return None
    values = np.array([r[field] for r in rows], dtype=np.float64)
    weights = np.array([r[weight] for r in rows], dtype=np.float64)
    months = [r["date"][:6] for r in rows]
    coins = [r["coin"] for r in rows]
    return cluster_bootstrap(values, weights, months, n_boot=n_boot, seed=seed,
                             cluster_b=coins)


def month_bootstrap_ci(rows, field, weight, n_boot=N_BOOT, seed=SEED):
    boot = month_bootstrap(rows, field, weight, n_boot, seed)
    return [float("nan"), float("nan")] if boot is None else boot["ci95"]


def horizon_saturation(rows):
    """Compare the 60s adverse move against the 10s one to see if it keeps growing."""
    a10 = pooled(rows, "adverse_select_bp_10s", "fills")
    a60 = pooled(rows, "adverse_select_bp_60s", "fills")
    return {
        "adverse_bp_10s": a10,
        "adverse_bp_60s": a60,
        "ratio_60s_over_10s": a60 / a10 if a10 else float("nan"),
        "ratio_if_drift_continued_at_the_same_rate": 6.0,
        "value_if_flow_carried_no_information": 0.0,
        "note": ("this term is a signed markout, so under a martingale mid and flow "
                 "that carries no information it is zero at every horizon rather "
                 "than growing with the square root of time, which is how the "
                 "unsigned magnitude would scale. The measured value is far from "
                 "zero and close to flat between the two horizons, so the adverse "
                 "move is substantially complete by ten seconds instead of "
                 "accumulating. Had it continued at the ten second rate the ratio "
                 "would be six. This compares a measured ratio against two "
                 "reference values and does not test permanence, and the capture "
                 "leg is invariant across the horizons by construction so it "
                 "carries none of the comparison"),
    }


def weighting_block(rows, tau, weight, suffix):
    cap = pooled(rows, f"spread_capture{suffix}_bp_{tau}", weight)
    adv = pooled(rows, f"adverse_select{suffix}_bp_{tau}", weight)
    net = pooled(rows, f"net_markout{suffix}_bp_{tau}", weight)
    boot = month_bootstrap(rows, f"net_markout{suffix}_bp_{tau}", weight)
    two = month_coin_two_way(rows, f"net_markout{suffix}_bp_{tau}", weight)
    ci = [float("nan"), float("nan")] if boot is None else boot["ci95"]
    return {
        "capture_bp": cap,
        "adverse_bp": adv,
        "net_bp": net,
        "adv_over_cap": abs(adv) / abs(cap) if cap else float("nan"),
        "net_ci95": ci,
        "net_clears_zero": (None if boot is None or not boot["verdict_is_available"]
                            else boot["clears_zero"]),
        "net_clears_zero_percentile": (
            None if boot is None or not boot["verdict_is_available"]
            else bool(ci[0] > 0 or ci[1] < 0)),
        "net_ci95_t": None if boot is None else boot["ci95_t"],
        "n_month_clusters": None if boot is None else boot["n_clusters"],
        "net_ci95_t_two_way_month_coin": (
            None if two is None else two["ci95_t_two_way"]),
        "net_se_month_only": None if boot is None else boot["se_cluster"],
        "net_se_two_way_month_coin": (
            None if two is None else two["se_used_for_verdict"]),
        "net_clears_zero_two_way": (
            None if two is None or not two["verdict_is_available"]
            else two["clears_zero"]),
        "n_effective_months_kish": (None if boot is None
                                    else boot["n_effective_clusters"]),
        "percentile_interval_has_power_to_fail": (None if boot is None
                                       else boot["percentile_interval_can_reach_zero"]),
    }


def per_coin_block(rows, tau="10s"):
    by = defaultdict(list)
    for r in rows:
        by[r["coin"]].append(r)
    out = {}
    for coin in sorted(by):
        cr = by[coin]
        net = pooled(cr, f"net_markout_bp_{tau}", "fills")
        boot = month_bootstrap(cr, f"net_markout_bp_{tau}", "fills")
        ci = [float("nan"), float("nan")] if boot is None else boot["ci95"]
        cit = [float("nan"), float("nan")] if boot is None else boot["ci95_t"]
        available = bool(boot is not None and boot["verdict_is_available"])
        out[coin] = {
            "n_coindays": len(cr),
            "n_fills": int(sum(r["fills"] for r in cr)),
            "capture_bp": pooled(cr, f"spread_capture_bp_{tau}", "fills"),
            "adverse_bp": pooled(cr, f"adverse_select_bp_{tau}", "fills"),
            "net_bp": net,
            "net_ci95": ci,
            "negative": bool(net < 0),
            "verdict_is_available": available,
            "net_ci95_t": cit,
            "negative_clears_zero": bool(cit[1] < 0) if available else None,
            "positive_clears_zero": bool(cit[0] > 0) if available else None,
            "negative_clears_zero_percentile": (bool(ci[1] < 0) if available
                                                else None),
            "positive_clears_zero_percentile": (bool(ci[0] > 0) if available
                                                else None),
        }
    return out


def leave_one_coin_out(rows, tau="10s"):
    coins = sorted({r["coin"] for r in rows})
    full = pooled(rows, f"net_markout_bp_{tau}", "fills")
    by_coin = defaultdict(list)
    for r in rows:
        by_coin[r["coin"]].append(r)
    own_opposite = [c for c in coins
                    if (pooled(by_coin[c], f"net_markout_bp_{tau}", "fills") < 0)
                    != (full < 0)]
    opposite = []
    swings = {}
    for coin in coins:
        kept = [r for r in rows if r["coin"] != coin]
        val = pooled(kept, f"net_markout_bp_{tau}", "fills")
        swings[coin] = val - full
        if (val < 0) != (full < 0):
            opposite.append(coin)
    ranked = sorted(swings, key=lambda c: abs(swings[c]), reverse=True)
    return {
        "pooled_net_bp": full,
        "swing_bp_by_coin": swings,
        "most_influential_coin": ranked[0] if ranked else None,
        "n_coins_opposite_sign_to_pooled": len(own_opposite),
        "coins_opposite_sign": own_opposite,
        "n_coins_whose_removal_flips_pooled_sign": len(opposite),
        "coins_whose_removal_flips_pooled_sign": opposite,
        "sign_stable": not opposite,
    }


def venue_block(rows):
    coins = sorted({r["coin"] for r in rows})
    dates = sorted({r["date"] for r in rows})
    block = {
        "n_coindays": len(rows),
        "n_coins": len(coins),
        "n_fills": int(sum(r["fills"] for r in rows)),
        "coins": coins,
        "date_first": dates[0],
        "date_last": dates[-1],
        "n_months": len({d[:6] for d in dates}),
        "pooled": {},
    }
    for tau in TAUS:
        block["pooled"][f"{tau}_row_mean"] = weighting_block(
            rows, tau, "fills", "")
        block["pooled"][f"{tau}_coin_unit_weighted"] = weighting_block(
            rows, tau, "size", "_szw")
    pc = per_coin_block(rows)
    block["per_coin"] = pc
    neg = [c for c, v in pc.items() if v["negative"]]
    block["n_coins_net_negative"] = len(neg)
    block["n_coins_net_negative_clearing_zero"] = sum(
        1 for v in pc.values() if v["negative_clears_zero"])
    block["coins_significantly_positive"] = sorted(
        c for c, v in pc.items() if v["positive_clears_zero"])
    block["counterexamples_to_net_le_zero"] = len(
        block["coins_significantly_positive"])
    block["leave_one_coin_out"] = leave_one_coin_out(rows)
    block["horizon_saturation"] = horizon_saturation(rows)
    return block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", default=str(REPRODUCE / "panels"))
    ap.add_argument("--out", required=True,
                    help="output path; pass a scratch file to compare against "
                         "the shipped decomposition_by_venue.json")
    a = ap.parse_args()
    result = {"venues": {}}
    for venue in VENUES:
        path = Path(a.panels) / f"{venue}_coindays.csv"
        rows = load(path)
        print(f"{venue:14s} {len(rows):>5} coin-days, "
              f"{len({r['coin'] for r in rows})} coins")
        result["venues"][venue] = venue_block(rows)
    Path(a.out).write_text(json.dumps(result, indent=2, default=float))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
