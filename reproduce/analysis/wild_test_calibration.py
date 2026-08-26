"""
Measure the sign-flip test's size on the month structure these panels carry.

Run from the repository root: python3 reproduce/analysis/wild_test_calibration.py
--out wild_test_calibration.rebuilt.json

The sign-flip test is exact only when cluster contributions are symmetric about
their mean. The coverage sweep stresses that with a lognormal cluster effect,
which is a shape chosen to break the test rather than a shape taken from the
data. This closes that gap by asking what the test does on the distribution the
panels actually have.

Each venue's months are collapsed to one fill-weighted net per month, centred so
the null holds, then resampled whole with replacement. The test is run on each
resample and the rejection rate is reported against a nominal five percent.
Skewness and the share of months falling on the median's side of the mean are
reported alongside, because the second is what drives the failure: a
right-skewed effect puts most clusters on one side, so the observed statistic
sits at the edge of its own sign-flip reference distribution.

No shipped result reads a p-value from this test. It is exercised by the smoke
test and by the coverage sweep and nothing else, so what follows bounds a
library property rather than qualifying a published verdict.
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

from makercex import wild_cluster_p

VENUES = ("bybit_perp", "binance_um", "hyperliquid")
NOMINAL = 0.05
N_RESAMPLE = 3000
SEED = 5


def month_contributions(venue):
    """One fill-weighted net markout per calendar month, with its total weight."""
    path = REPRODUCE / "panels" / f"{venue}_coindays.csv"
    acc = defaultdict(lambda: [0.0, 0.0])
    with open(path) as fh:
        for r in csv.DictReader(fh):
            w = float(r["n_fills"])
            acc[r["date"][:6]][0] += float(r["net_markout_bp_10s"]) * w
            acc[r["date"][:6]][1] += w
    months = sorted(acc)
    value = np.array([acc[m][0] / acc[m][1] for m in months])
    weight = np.array([acc[m][1] for m in months])
    return value, weight


def shape(value):
    centred = value - value.mean()
    sd = centred.std(ddof=1)
    skew = float((centred ** 3).mean() / sd ** 3) if sd > 0 else float("nan")
    side = float(np.mean(np.sign(centred) == np.sign(np.median(centred))))
    return skew, side


def size_on_real_months(value, weight, n_resample=N_RESAMPLE, seed=SEED):
    g = value.size
    centred = value - np.average(value, weights=weight)
    rng = np.random.default_rng(seed)
    ran = rejected = 0
    labels = [str(i) for i in range(g)]
    for _ in range(n_resample):
        pick = rng.integers(0, g, g)
        p = wild_cluster_p(centred[pick], weight[pick], labels)
        if np.isfinite(p["p"]):
            ran += 1
            rejected += int(p["p"] < NOMINAL)
    return (rejected / ran if ran else float("nan")), ran


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-resample", type=int, default=N_RESAMPLE)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {"nominal": NOMINAL, "n_resample": a.n_resample, "venues": {}}
    for venue in VENUES:
        value, weight = month_contributions(venue)
        skew, side = shape(value)
        size, ran = size_on_real_months(value, weight, a.n_resample)
        out["venues"][venue] = {
            "n_months": int(value.size),
            "cross_month_skew": skew,
            "share_of_months_on_the_median_side": side,
            "sign_flip_size_on_resampled_months": size,
            "n_resamples_returning_a_p_value": ran,
            "exceeds_nominal_by": size / NOMINAL,
        }
        print(f"  {venue:12s} months {value.size:>2}  skew {skew:+.3f}  "
              f"median side {side:.2f}  size {size:.4f}")
    out["no_shipped_verdict_uses_this_test"] = True
    out["limitations"] = {
        "resampling_months_cannot_exceed_their_own_range": (
            "The reference distribution is built by resampling whole months, so "
            "it inherits whatever shape those months have and cannot represent a "
            "venue whose true month distribution is worse than the sample drawn."),
        "one_horizon_and_one_weighting": (
            "Months are collapsed on the 10 second net markout weighted by fill "
            "count, matching the pooled table. A different horizon or weighting "
            "would give a different month distribution."),
    }
    Path(a.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
