"""
Rebuild the sign-rule counterfactual: what a classifier costs a markout study.

Run from the repository root: python reproduce/analysis/sign_rule_counterfactual.py
--out sign_rule_counterfactual.rebuilt.json

Every maker-markout study on futures or equities has to infer the aggressor
side from price and volume, and none of them can price the error that
introduces, because none of them hold the true sign to check against. These
venues publish it. The panel behind this producer is the identical
decomposition run three times on the identical tape, once on the exchange
aggressor flag and once under each of the two classifiers the literature
actually uses, so the error is a measured quantity here rather than an assumed
one.

The sign is not a label on a fixed set of fills, which is why the whole
simulation is rerun under each rule rather than the fills relabelled. It decides
which side of the book a trade lifts, so it decides which of the maker's two
resting orders fills at all, and only then does it enter the markout as the
maker's side. A researcher holding a classifier and nothing else gets a
different fill set and a different sign on each fill, and both channels are in
the numbers below. The fill-count column is reported for exactly that reason:
where a rule's fill count moves, part of its markout error is a change in what
was measured and not only in how it was signed.

Two numbers per venue per rule carry the result. Sign accuracy is the share of
trades the rule signs the way the exchange did, trade-weighted, with the
size-weighted twin beside it because the two can differ when the misclassified
trades are the large ones. Net error is the rule's pooled net entry markout
minus the truth's, so a positive error means the classifier reports a less
negative number than the exchange flag supports.

The interval on that error is a paired one. The same coin-day appears under
every rule, so the difference is taken within the coin-day first and the
interval is a two-way cluster bootstrap on calendar month and coin over those
differences, which is the estimator the paper's headline uses. Pairing matters:
the two rules' pooled figures are each dominated by the same handful of
high-volume coins, and an unpaired interval would price that shared variation
twice.

All three venues carry this analysis. The Binance panel is measured by
streaming its daily archive rather than from a local store, which is why it
arrived after the other two, and its ``true`` rows reproduce the shipped Binance
panel on all 4,785 coin-days.

One lookup decides whether any of this is comparable. Lee-Ready needs the quote
that stood when the order arrived, and a book update caused by a trade often
carries that trade's own timestamp, so the last quote AT OR BEFORE the trade is
the post-trade book and inverts the classification on exactly the price-moving
trades. The damage scales with the feed's clock: trades sharing a timestamp with
a book update run 0.3 percent on Bybit, 8 to 10 on Hyperliquid and 52.7 on
millisecond-stamped Binance. Taking the quote strictly before the trade is both
the definition and the only choice that compares venues rather than clocks.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

HERE = Path(__file__).resolve().parent
REPRODUCE = HERE.parent
sys.path.insert(0, str(REPRODUCE.parent))

from makercex import cluster_bootstrap

PANEL = REPRODUCE / "panels" / "sign_rule_coindays.csv"
RULES = ("true", "lee_ready", "tick")
CLASSIFIERS = ("lee_ready", "tick")
N_BOOT = 4000
SEED = 313

RULE_LABEL = {"true": "exchange aggressor flag", "lee_ready": "Lee-Ready",
              "tick": "tick rule"}

# Venue order is fixed here rather than sorted, so the result file, the figure
# and every table in the paper read the same way round. Alphabetical would put
# Binance first and disagree with the coverage and headline tables.
VENUE_ORDER = ("bybit_perp", "binance_um", "hyperliquid")


def load(path):
    rows = collections.defaultdict(dict)
    with Path(path).open(newline="") as handle:
        for r in csv.DictReader(handle):
            rows[(r["venue"], r["coin"], r["date"])][r["rule"]] = r
    complete = {k: v for k, v in rows.items() if all(rule in v for rule in RULES)}
    return complete, len(rows) - len(complete)


def _f(row, key):
    value = row.get(key, "")
    return float(value) if value not in ("", None) else float("nan")


def pooled(cells, rule, key, weight="n_fills"):
    w = np.array([_f(c[rule], weight) for c in cells])
    v = np.array([_f(c[rule], key) for c in cells])
    good = np.isfinite(w) & np.isfinite(v) & (w > 0)
    if not good.any():
        return float("nan")
    return float((v[good] * w[good]).sum() / w[good].sum())


def accuracy(cells, rule):
    """Trade-weighted and size-weighted share of trades signed as the venue did."""
    n = np.array([_f(c[rule], "n_trades") for c in cells])
    a = np.array([_f(c[rule], "sign_accuracy") for c in cells])
    s = np.array([_f(c[rule], "sign_accuracy_szw") for c in cells])
    good = np.isfinite(n) & np.isfinite(a) & (n > 0)
    trade_w = float((a[good] * n[good]).sum() / n[good].sum())
    good_s = good & np.isfinite(s)
    size_w = (float((s[good_s] * n[good_s]).sum() / n[good_s].sum())
              if good_s.any() else float("nan"))
    return trade_w, size_w


def paired_error(cells, rule, key, horizon_weight="n_fills"):
    """Per-coin-day (rule minus truth), with a two-way month-and-coin interval.

    The weight is the truth's fill count on that coin-day, held fixed across the
    rules so that a rule taking more or fewer fills does not also reweight the
    panel underneath its own error.
    """
    diffs, weights, months, coins = [], [], [], []
    for (venue, coin, date), by_rule in cells:
        d = _f(by_rule[rule], key) - _f(by_rule["true"], key)
        w = _f(by_rule["true"], horizon_weight)
        if not (np.isfinite(d) and np.isfinite(w) and w > 0):
            continue
        diffs.append(d); weights.append(w)
        months.append(f"{date[:4]}-{date[4:6]}"); coins.append(coin)
    if not diffs:
        return None
    return cluster_bootstrap(np.array(diffs), np.array(weights), months,
                             n_boot=N_BOOT, seed=SEED, cluster_b=coins)


def per_coin_accuracy(items, rule):
    """Trade-weighted accuracy for one rule, coin by coin.

    The pooled figure hides most of what a reader needs here: agreement runs
    with the tick grid, so the coin-level spread is wider than the venue-level
    number and is the part that transfers to a different coin list.
    """
    hit = collections.defaultdict(float)
    seen = collections.defaultdict(float)
    for (venue, coin, date), by_rule in items:
        n = _f(by_rule[rule], "n_trades")
        a = _f(by_rule[rule], "sign_accuracy")
        if not (np.isfinite(n) and np.isfinite(a) and n > 0):
            continue
        hit[coin] += a * n
        seen[coin] += n
    return {c: hit[c] / seen[c] for c in sorted(seen)}


def venue_block(items):
    cells = [v for _, v in items]
    out = {"n_coindays": len(cells), "by_rule": {}, "net_error_10s": {},
           "net_error_60s": {}}
    for rule in RULES:
        trade_w, size_w = accuracy(cells, rule)
        out["by_rule"][rule] = {
            "label": RULE_LABEL[rule],
            "sign_accuracy": trade_w,
            "sign_accuracy_size_weighted": size_w,
            "n_fills": int(sum(_f(c[rule], "n_fills") for c in cells)),
            "capture_bp_10s": pooled(cells, rule, "spread_capture_bp_10s"),
            "adverse_bp_10s": pooled(cells, rule, "adverse_select_bp_10s"),
            "net_bp_10s": pooled(cells, rule, "net_markout_bp_10s"),
            "capture_bp_60s": pooled(cells, rule, "spread_capture_bp_60s"),
            "adverse_bp_60s": pooled(cells, rule, "adverse_select_bp_60s"),
            "net_bp_60s": pooled(cells, rule, "net_markout_bp_60s"),
        }
    truth = out["by_rule"]["true"]
    for rule in CLASSIFIERS:
        out["by_rule"][rule]["sign_accuracy_by_coin"] = per_coin_accuracy(items, rule)
    for rule in CLASSIFIERS:
        got = out["by_rule"][rule]
        got["net_error_bp_10s"] = got["net_bp_10s"] - truth["net_bp_10s"]
        got["net_error_bp_60s"] = got["net_bp_60s"] - truth["net_bp_60s"]
        got["capture_error_bp_10s"] = got["capture_bp_10s"] - truth["capture_bp_10s"]
        got["adverse_error_bp_10s"] = got["adverse_bp_10s"] - truth["adverse_bp_10s"]
        got["fill_count_ratio"] = (got["n_fills"] / truth["n_fills"]
                                   if truth["n_fills"] else float("nan"))
        got["net_error_as_share_of_truth"] = (
            got["net_error_bp_10s"] / abs(truth["net_bp_10s"])
            if truth["net_bp_10s"] else float("nan"))
        for tau, key in (("10s", "net_markout_bp_10s"), ("60s", "net_markout_bp_60s")):
            ci = paired_error(items, rule, key)
            out[f"net_error_{tau}"][rule] = ci
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=str(PANEL))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cells, dropped = load(a.panel)
    venues = collections.defaultdict(list)
    for key, by_rule in cells.items():
        venues[key[0]].append((key, by_rule))

    out = {
        "panel": Path(a.panel).name,
        "n_boot": N_BOOT,
        "seed": SEED,
        "rules": {r: RULE_LABEL[r] for r in RULES},
        "n_coindays_dropped_for_incomplete_rule_set": dropped,
        "venues": {v: venue_block(sorted(venues[v]))
                   for v in VENUE_ORDER if v in venues},
        "scope": {
            "binance_is_the_untested_cell": (
                "Binance carries no fill ledger and no local trade archive, so "
                "this counterfactual is run on Bybit and Hyperliquid only. The "
                "third venue is untested here rather than tested and agreeing."),
            "the_classifier_changes_the_fill_set_too": (
                "A sign decides which of the maker's two resting orders a trade "
                "can lift, so each rule takes its own fill set. The fill count "
                "ratio is reported per rule because part of a rule's markout "
                "error is a change in which fills were measured."),
            "accuracy_is_against_the_venue_flag": (
                "Accuracy is agreement with the exchange aggressor flag, which "
                "is taken as ground truth throughout. Where a venue's own flag "
                "is wrong this measures agreement rather than correctness."),
            "one_quoter_one_cadence": (
                "Every figure is the reference last-in-queue touch quoter at the "
                "same 100 ms re-quote ceiling used everywhere else in the paper. "
                "The error a classifier carries for a different quoter or a "
                "different cadence is not measured here."),
        },
    }
    Path(a.out).write_text(json.dumps(out, indent=2, default=float) + "\n")

    for venue, block in out["venues"].items():
        truth = block["by_rule"]["true"]
        print(f"\n{venue}: {block['n_coindays']:,} coin-days, "
              f"truth net {truth['net_bp_10s']:+.4f} bp on "
              f"{truth['n_fills']:,} fills")
        for rule in CLASSIFIERS:
            got = block["by_rule"][rule]
            ci = block["net_error_10s"][rule]
            verdict = ("clears zero" if ci and ci["clears_zero"]
                       else "does not clear" if ci and ci["verdict_is_available"]
                       else "withheld")
            lo, hi = (ci["ci95_t_two_way"] or ci["ci95_t"]) if ci else (float("nan"),) * 2
            print(f"  {RULE_LABEL[rule]:<24s} accuracy {got['sign_accuracy']:.4f}  "
                  f"net {got['net_bp_10s']:+.4f}  error {got['net_error_bp_10s']:+.4f} "
                  f"[{lo:+.4f}, {hi:+.4f}] {verdict}  "
                  f"fills x{got['fill_count_ratio']:.3f}")


if __name__ == "__main__":
    main()
