"""
Bound the survivorship bias on the Bybit and Binance panels.

Run from the repository root:
python reproduce/analysis/survivorship_bound.py --out survivorship_bound.rebuilt.json

The three shipped panels contain no delisted coin: every coin in each panel
trades on that panel's last sampled date. The universe behind the headline is
therefore a survivor set, and this script turns the direction of that omission
from an argument into a number, following the plan hashed in
`reproduce/preregistration/survivorship_bound.md`.

Matched dates are the whole design. A delisted coin has data up to its death
while survivors run to the end of the window, so a raw contrast between the two
would compare 2023 against 2024 and 2025 rather than dying against surviving.
Every contrast here is taken within a date: a delisted coin-day is differenced
against the fill-weighted net of the live panel on that same date, so the live
side is reweighted onto the dead coin's own calendar and the year drops out.

The contrast series is per coin-day, clustered two-way on month and coin
through the shipped library, and the library's Kish floor applies unchanged: a
cell below five effective clusters returns no verdict rather than a weak one.
Individual coins are expected to abstain, since a coin that traded forty days
spans two months. The pooled gap is the number that carries and the per-coin
column is dispersion.

The bound multiplies a death rate by that gap and is computed under two
weightings, because the answers differ by construction and picking one would be
picking the flattering one. The coin-count reading treats every name as a unit
and uses the venue-wide death rate. The fill-weight reading uses the share of
simulated fills the delisted coins carry over their matched dates, and its
numerator and denominator both cover the measured set rather than the whole
cross-section, which is recorded in the output rather than left to be assumed.

Commensurability of the delisted rows with the shipped panels is established in
the measurement stage, not here. The raw venue archives sit outside this
repository by the boundary stated in the paper, so a re-measurement check
cannot run inside the package; its result is carried in
`commensurability` below and reproduced by `scripts/surv/validate.py` in the
measurement tree, where the archives are reachable.

Three details of the implementation are worth stating once here rather than at
their call sites.

**Power to fail.** Where the monthly means behind a gap share a sign, a resample
of whole months cannot produce a pooled value on the other side of zero, so the
percentile interval's clears-zero flag is an identity rather than a test. That
is the reading the per-coin surface already refuses, and it is refused here too:
`percentile_interval_can_reach_zero` travels with every verdict, and where it is
false the evidence for the gap is the unanimity across coins, not the interval.

**Abstention on a non-finite interval.** A second clustering dimension with two
groups leaves the two-way t on zero degrees of freedom, which yields a non-finite
bound while the library still reports the verdict as available. No shipped cell
reaches that state, so it is handled here rather than by changing the library
underneath every other result that depends on it: a cell whose verdict interval
is not finite abstains, for the same reason the Kish floor abstains one dimension
up.

**Worst case against central case.** A bound should be quoted from the far end of
its interval, so `bound_bp_*_worst_case` reads the verdict interval and the
unsuffixed fields read the point estimate. That distinction is what keeps a venue
whose gap is imprecise, but whose death rate is small, from looking unbounded.
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
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from makercex.inference import (MIN_CLUSTERS, cluster_bootstrap,  # noqa: E402
                                effective_clusters)

PANELS = ROOT / "reproduce" / "panels"
DELISTED = Path(os.environ.get("SURV_DELISTED_PANEL",
                               PANELS / "delisted_coindays.csv"))

# Venue-wide death rates, from the archive counts frozen in the plan.
DEATHS = {
    "bybit_perp": {"panel": "bybit_perp_coindays.csv", "suffix": "USDT",
                   "listed_at_start": 206, "delisted_in_window": 30,
                   "window": ["2023-04-01", "2025-08-18"]},
    "binance_um": {"panel": "binance_um_coindays.csv", "suffix": "",
                   "listed_at_start": 185, "delisted_in_window": 4,
                   "window": ["2023-05-16", "2024-03-29"]},
}

# Contracts on index products rather than on a single coin. Reported separately
# because a basket is not the object the rest of the paper measures.
INDEX_PRODUCTS = {"BLUEBIRDUSDT", "FOOTBALLUSDT"}

TAUS = ("10s", "60s")


def read_rows(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def fnum(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def thin_live_coins(panel_rows, frac=1 / 3):
    """The least active third of the live panel, by total fills.

    POST-LOCK ADDITION, not in the hashed plan, and labelled as such wherever it
    is reported. The delisted coins are thin: on Bybit their captured half-spread
    runs an order of magnitude above the live panel's, which is a property of a
    wide tick-to-price ratio and not of dying. Matching on date fixes the
    calendar and leaves that confound untouched, so the whole-panel gap below
    prices dying and thinness together. Benchmarking the same delisted coin-days
    against only the least active live coins holds liquidity tier roughly fixed
    and separates the two. The whole-panel gap remains the pre-registered
    primary; this is reported beside it, never in place of it.
    """
    # Mean fills per coin-day, not total. Ranking on the total would call a
    # coin thin for being late to the panel: HYPE carries 90 of Bybit's 295
    # sampled dates because it listed in December 2024, and it is one of that
    # venue's more active names on the days it does trade.
    tot = defaultdict(float)
    days = defaultdict(int)
    for r in panel_rows:
        w = fnum(r, "n_fills")
        if np.isfinite(w):
            tot[r["coin"]] += w
            days[r["coin"]] += 1
    rate = {c: tot[c] / days[c] for c in tot if days[c]}
    order = sorted(rate, key=lambda c: rate[c])
    k = max(1, int(round(len(order) * frac)))
    return set(order[:k])


def live_daily_net(panel_rows, tau, only=None):
    """Fill-weighted net of the live panel, per date. The dead coin's benchmark."""
    num = defaultdict(float)
    den = defaultdict(float)
    for r in panel_rows:
        if only is not None and r["coin"] not in only:
            continue
        w = fnum(r, "n_fills")
        v = fnum(r, f"net_markout_bp_{tau}")
        if np.isfinite(w) and np.isfinite(v) and w > 0:
            num[r["date"]] += v * w
            den[r["date"]] += w
    return {d: num[d] / den[d] for d in num if den[d] > 0}, den


def fills_per_day(rows, key, subset=None):
    tot = n = 0.0
    for r in rows:
        if subset is not None and r[key] not in subset:
            continue
        w = fnum(r, "n_fills")
        if np.isfinite(w):
            tot += w
            n += 1
    return tot / n if n else float("nan")


def contrasts(dead_rows, live_by_date, tau):
    """One contrast per delisted coin-day: dead net minus that date's live net."""
    vals, wts, months, coins, dates = [], [], [], [], []
    for r in dead_rows:
        d = r["date"]
        if d not in live_by_date:
            continue
        w = fnum(r, "n_fills")
        v = fnum(r, f"net_markout_bp_{tau}")
        if not (np.isfinite(w) and np.isfinite(v) and w > 0):
            continue
        vals.append(v - live_by_date[d])
        wts.append(w)
        months.append(d[:6])
        coins.append(r["symbol"])
        dates.append(d)
    return (np.array(vals), np.array(wts), months, coins, dates)


def weighted(vals, wts):
    return float((vals * wts).sum() / wts.sum()) if wts.sum() > 0 else float("nan")


def venue_block(venue, spec, dead_all, n_boot, seed):
    live_rows = read_rows(PANELS / spec["panel"])
    dead_rows = [r for r in dead_all if r["venue"] == venue]
    out = {"n_delisted_coins": len({r["symbol"] for r in dead_rows}),
           "n_delisted_coindays": len(dead_rows),
           "listed_at_window_start": spec["listed_at_start"],
           "delisted_in_window": spec["delisted_in_window"],
           "death_rate_by_coin": spec["delisted_in_window"] / spec["listed_at_start"],
           "window": spec["window"], "by_horizon": {}}

    thin = thin_live_coins(live_rows)
    out["thin_live_benchmark_coins"] = sorted(thin)
    for tau in TAUS:
        full_by_date, live_fills_by_date = live_daily_net(live_rows, tau)
        thin_by_date, _ = live_daily_net(live_rows, tau, only=thin)
        subsets = {"all": (dead_rows, full_by_date)}
        if venue == "binance_um":
            subsets["single_coin_only"] = (
                [r for r in dead_rows if r["symbol"] not in INDEX_PRODUCTS],
                full_by_date)
        subsets["all_vs_thin_live"] = (dead_rows, thin_by_date)
        block = {}
        for label, (rows, bench_by_date) in subsets.items():
            vals, wts, months, coins, dates = contrasts(rows, bench_by_date, tau)
            if vals.size == 0:
                block[label] = None
                continue
            ci = cluster_bootstrap(vals, wts, months, n_boot=n_boot, seed=seed,
                                   cluster_b=coins)
            # See "Abstention on a non-finite interval" in the module docstring.
            bounds = ci["ci95_t_two_way"] if ci["verdict_is_two_way"] else ci["ci95_t"]
            if bounds is None or not all(np.isfinite(b) for b in bounds):
                ci = dict(ci, verdict_is_available=False, clears_zero=False,
                          verdict_interval="none: two-way t on zero df")
            # per-coin dispersion: each coin's own fill-weighted contrast
            per_coin = {}
            for c in sorted(set(coins)):
                m = np.array([x == c for x in coins])
                per_coin[c] = {
                    "gap_bp": weighted(vals[m], wts[m]),
                    "n_coindays": int(m.sum()),
                    "n_fills": int(wts[m].sum()),
                    "n_months": len({d[:6] for d, k in zip(dates, coins) if k == c}),
                }
            gap_fill_wtd = weighted(vals, wts)
            gap_by_coin = float(np.mean([v["gap_bp"] for v in per_coin.values()]))
            dead_fills = float(wts.sum())
            matched_live_fills = float(sum(live_fills_by_date[d]
                                           for d in sorted(set(dates))))
            fill_share = dead_fills / (dead_fills + matched_live_fills)
            block[label] = {
                "gap_bp_fill_weighted": gap_fill_wtd,
                "gap_bp_mean_over_coins": gap_by_coin,
                "gap_ci95_two_way_month_coin": ci["ci95_t_two_way"],
                "gap_ci95_t_month_only": ci["ci95_t"],
                "gap_ci95_percentile_month": ci["ci95"],
                # Which interval the verdict was actually read off. A two-way
                # variance cannot be formed from a single coin cluster, and in
                # that case the library falls back to the month-only t interval
                # rather than returning nothing; saying so here keeps a null
                # two-way bound beside a live verdict from reading as a defect.
                "gap_verdict_interval": ci["verdict_interval"],
                "gap_verdict_is_two_way": bool(ci["verdict_is_two_way"]),
                "gap_verdict_is_available": bool(ci["verdict_is_available"]),
                "gap_clears_zero": bool(ci["clears_zero"]),
                # See "Power to fail" in the module docstring.
                "percentile_interval_can_reach_zero": bool(
                    ci["percentile_interval_can_reach_zero"]),
                "n_coins_negative": sum(1 for v in per_coin.values()
                                        if v["gap_bp"] < 0),
                "n_coins": len(per_coin),
                "gap_clears_zero_percentile": bool(ci["clears_zero_percentile"]),
                "n_month_clusters": int(ci["n_clusters"]),
                "n_effective_months_kish": float(ci["n_effective_clusters"]),
                "min_effective_clusters_floor": MIN_CLUSTERS,
                "n_coindays": int(vals.size),
                "n_dead_fills": dead_fills,
                "n_matched_live_fills": matched_live_fills,
                "dead_fill_share_of_measured_set": fill_share,
                "bound_bp_by_coin_count": out["death_rate_by_coin"] * gap_by_coin,
                "bound_bp_by_fill_weight": fill_share * gap_fill_wtd,
                # "at most", read off the interval rather than the point.
                "bound_bp_by_coin_count_worst_case": (
                    out["death_rate_by_coin"] * max(abs(b) for b in bounds)
                    * (-1 if gap_by_coin < 0 else 1)
                    if bounds and all(np.isfinite(b) for b in bounds) else None),
                "benchmark": ("least active third of the live panel"
                              if label.endswith("thin_live") else "whole live panel"),
                "is_preregistered": not label.endswith("thin_live"),
                # The residual tier gap the benchmark does not close. Even the
                # least active live coins are far busier than the delisted set,
                # so the thin-live contrast narrows the liquidity confound
                # rather than removing it, and the ratio says by how much.
                "dead_fills_per_coinday": fills_per_day(rows, "symbol"),
                "benchmark_fills_per_coinday": (
                    fills_per_day(live_rows, "coin",
                                  thin if label.endswith("thin_live") else None)),
                "per_coin": per_coin,
            }
        out["by_horizon"][tau] = block
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-boot", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if not DELISTED.exists():
        raise SystemExit(f"missing {DELISTED}; run the measurement stage first")
    dead_all = read_rows(DELISTED)

    report = {
        "scope": ("Bybit and Binance USD-M only. Hyperliquid is out of scope: both "
                  "of its archive buckets are requester-pays and no listing or "
                  "delisting record was obtainable."),
        "design": ("Contrast is per delisted coin-day against the live panel's "
                   "fill-weighted net on the same date, so the live side is "
                   "reweighted onto the dead coin's calendar. Interval is two-way "
                   "clustered on month and coin with the library's Kish floor."),
        "preregistration": "reproduce/preregistration/survivorship_bound.md",
        "selection_rule": ("The ten delisted perpetuals with the most in-window "
                           "trading days, which are the longest-lived and so the "
                           "least distressed of the qualifying set. The gap measured "
                           "on them is expected to understate the gap across all "
                           "qualifying delistings, making the bound an "
                           "understatement rather than a conservative overstatement."),
        "commensurability": {
            "checked_in": "scripts/surv/validate.py (measurement tree)",
            "bybit_coindays_reproducing_shipped_panel": "16 of 16",
            "note": ("The delisted rows are measured on the shipped Bybit panel's "
                     "publish-time book basis, not the engine stamp the order-book "
                     "parser now prefers, so the contrast carries no book-time "
                     "shift. Binance bookTicker has a single causal stamp and no "
                     "such choice arises."),
        },
        "venues": {},
    }
    for venue, spec in DEATHS.items():
        report["venues"][venue] = venue_block(venue, spec, dead_all,
                                              args.n_boot, args.seed)

    Path(args.out).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    for venue, blk in report["venues"].items():
        b = blk["by_horizon"]["10s"]["all"]
        if b:
            print(f"{venue}: gap {b['gap_bp_fill_weighted']:+.4f} bp fill-wtd, "
                  f"{b['gap_bp_mean_over_coins']:+.4f} by coin; "
                  f"bound {b['bound_bp_by_coin_count']:+.4f} (coin count) / "
                  f"{b['bound_bp_by_fill_weight']:+.4f} (fill weight)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
