"""
Smoke test on synthetic data, exercising the paper's mechanism end to end.

Run from the repository root: python examples/run_synthetic.py

The paper measures that a passive maker's captured half-spread is more than
consumed by post-fill adverse drift, so net entry markout is negative before fees.
This script builds a synthetic tape whose only free parameter is how informed
the incoming flow is, shows that raising it drives adverse selection past the
captured half-spread and asserts the identities and signs the decomposition has
to satisfy. It then runs the cluster-robust machinery on a synthetic panel so
the interval and power diagnostics can be checked end to end. Each check
records its result rather than raising, and the script exits non-zero if any
failed, so a clean exit means the library behaved as specified.

The tape is deliberately minimal: there is no queue, no partial fills, no
cancels and the markout is a single step ahead. It reproduces the sign and the
direction of the mechanism, not the magnitudes in the paper, which come from
the venue panels.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from makercex import (breakeven_rebate_bp, cluster_bootstrap, decompose,
                      make_panel, make_tape, net_after_fee, sign_test_p,
                      t_crit_95, two_way_se, wild_cluster_p)
from makercex import (MIN_CLUSTERS, benjamini_hochberg, simulate_touch_fills,
                      t_two_sided_p)

CHECKS = []
SPARSE_SEED = 1


def check(name, ok):
    CHECKS.append((name, bool(ok)))
    print(f"   {'pass' if ok else 'FAIL'}  {name}")
    return bool(ok)


def part_one():
    print("1. Adverse selection against the share of informed flow")
    print(f"   {'informed':>10} {'capture':>9} {'adverse':>9} {'net':>9} "
          f"{'adv/cap':>8}")
    nets = []
    for informed in (0.50, 0.60, 0.70, 0.80, 0.90):
        t = make_tape(n_events=40000, half_spread_bp=1.0, informed=informed,
                      seed=7)
        d = decompose(t["fill_price"], t["mid_at_fill"], t["mid_at_horizon"],
                      t["side"], t["size"])
        nets.append(d["net_bp"])
        print(f"   {informed:>10.2f} {d['capture_bp']:>9.4f} "
              f"{d['adverse_bp']:>9.4f} {d['net_bp']:>9.4f} "
              f"{d['adverse_over_capture']:>8.3f}")
    print("\n   Checks on the decomposition")
    check("net falls monotonically in the informed share",
          all(nets[i] > nets[i + 1] for i in range(len(nets) - 1)))
    check("net is negative once flow is mostly informed", nets[-1] < 0)
    return nets


def part_one_identities():
    t = make_tape(n_events=20000, half_spread_bp=1.4, informed=0.85, seed=11)
    size = np.abs(np.random.default_rng(2).standard_normal(20000)) + 0.1
    d = decompose(t["fill_price"], t["mid_at_fill"], t["mid_at_horizon"],
                  t["side"], size)
    pf = d["per_fill"]
    check("per-fill net equals per-fill capture plus adverse",
          np.allclose(pf["net_bp"], pf["capture_bp"] + pf["adverse_bp"]))
    check("aggregate net equals the size-weighted mean of per-fill net",
          abs(d["net_bp"] - float(np.average(pf["net_bp"], weights=size)))
          < 1e-9)
    check("aggregate net equals aggregate capture plus adverse",
          abs(d["net_bp"] - (d["capture_bp"] + d["adverse_bp"])) < 1e-9)
    independent = float(np.average(
        t["side"] * (t["mid_at_fill"] - t["fill_price"])
        / t["mid_at_fill"] * 1e4, weights=size))
    check("captured half-spread matches an independent computation",
          abs(d["capture_bp"] - independent) < 1e-9)
    check("captured half-spread recovers the quoted half-spread",
          abs(d["capture_bp"] - 1.4) < 0.01)
    check("adverse selection is negative under informed flow",
          d["adverse_bp"] < 0)
    check("captured half-spread is non-negative on genuinely passive fills",
          d["capture_bp"] >= 0 and bool((pf["capture_bp"] >= 0).all()))

    t_un = make_tape(n_events=20000, half_spread_bp=1.4, informed=0.15,
                     seed=11)
    d_un = decompose(t_un["fill_price"], t_un["mid_at_fill"],
                     t_un["mid_at_horizon"], t_un["side"])
    check("adverse selection flips sign when flow is uninformed",
          d_un["adverse_bp"] > 0)

    d_eq = decompose(t["fill_price"], t["mid_at_fill"], t["mid_at_horizon"],
                     t["side"])
    check("size weighting changes the answer",
          abs(d_eq["net_bp"] - d["net_bp"]) > 1e-6)

    mid_nan = np.array(t["mid_at_horizon"], dtype=float)
    mid_nan[:100] = np.nan
    d_nan = decompose(t["fill_price"], t["mid_at_fill"], mid_nan, t["side"])
    check("non-finite markouts are excluded from the fill count",
          d_nan["n_fills"] == d_eq["n_fills"] - 100)
    check("non-finite markouts do not poison the aggregates",
          all(np.isfinite(d_nan[k])
              for k in ("capture_bp", "adverse_bp", "net_bp")))

    check("breakeven rebate is the negated net",
          abs(breakeven_rebate_bp(d["net_bp"]) + d["net_bp"]) < 1e-12)
    check("a maker fee lowers net one for one",
          abs(net_after_fee(d["net_bp"], 0.5) - (d["net_bp"] - 0.5)) < 1e-12)


def coin_means_of(values, coins):
    return [values[[i for i, c in enumerate(coins) if c == u]].mean()
            for u in sorted(set(coins))]


def part_two():
    print("\n2. Cluster-robust inference on a synthetic panel")
    panel = make_panel(n_coins=8, n_dates=10, n_events=3000, seed=3,
                       informed_low=0.88, informed_high=0.96)
    nets, weights, coins, dates = [], [], [], []
    for row in panel:
        d = decompose(row["fill_price"], row["mid_at_fill"],
                      row["mid_at_horizon"], row["side"], row["size"])
        nets.append(d["net_bp"])
        weights.append(d["n_fills"])
        coins.append(row["coin"])
        dates.append(row["date"])
    nets = np.asarray(nets)
    boot = cluster_bootstrap(nets, weights, coins, n_boot=800, seed=5)
    tw = two_way_se(nets, weights, coins, dates)
    wild = wild_cluster_p(nets, weights, coins, n_boot=800, seed=5)
    n_pos = int((nets > 0).sum())
    print(f"   {len(nets)} coin-days over {len(set(coins))} coins and "
          f"{len(set(dates))} dates")
    print(f"   point {boot['point']:+.4f}  coin-clustered CI "
          f"[{boot['ci95'][0]:+.4f}, {boot['ci95'][1]:+.4f}]")
    print(f"   effective clusters {boot['n_effective_clusters']:.2f} of "
          f"{boot['n_clusters']}")
    print(f"   interval could reach zero: {boot['percentile_interval_can_reach_zero']}")
    print(f"   effective clusters {boot['n_effective_clusters']:.2f}, "
          f"t df {boot['t_degrees_of_freedom']}")
    print(f"   t interval [{boot['ci95_t'][0]:+.4f}, {boot['ci95_t'][1]:+.4f}]"
          f"  clears zero {boot['clears_zero']} "
          f"(percentile would say {boot['clears_zero_percentile']})")
    print(f"   one-way SE {tw['se_a']:.4f}  two-way SE {tw['se_two_way']:.4f}")
    print(f"   wild cluster p {wild['p']:.4f} "
          f"(exact enumeration: {wild['is_exact_enumeration']})")
    n_coin_pos = sum(1 for m in coin_means_of(nets, coins) if m > 0)
    n_coins = len(set(coins))
    print(f"   sign test on {n_coin_pos}/{n_coins} coins positive: "
          f"{sign_test_p(n_coin_pos, n_coins):.4f}")
    print(f"   the same test run on {len(nets)} coin-days would give "
          f"{sign_test_p(n_pos, len(nets)):.2e}, which treats coin-days as")
    print("   independent and is the error this module exists to avoid")

    print("\n   Checks on the inference machinery")
    coin_means = coin_means_of(nets, coins)
    unanimous = all(m < 0 for m in coin_means) or all(m > 0 for m in coin_means)
    check("the power flag matches whether the cluster means are unanimous",
          boot["percentile_interval_can_reach_zero"] is not unanimous)
    check("the point estimate lies inside its own interval",
          boot["ci95"][0] <= boot["point"] <= boot["ci95"][1])
    check("effective clusters never exceed the raw cluster count",
          boot["n_effective_clusters"] <= boot["n_clusters"] + 1e-9)
    check("the two-way SE is positive and finite",
          np.isfinite(tw["se_two_way"]) and tw["se_two_way"] > 0)
    check("the two-way correction actually moves the SE",
          abs(tw["se_two_way"] - tw["se_a"]) > 1e-9)
    check("eight clusters are enumerated exactly rather than sampled",
          wild["is_exact_enumeration"])
    check("the wild p respects its enumerable floor",
          wild["p"] >= 2.0 / 2 ** len(set(coins)) - 1e-12)

    noise = np.random.default_rng(9).standard_normal(len(nets)) * nets.std()
    w_noise = wild_cluster_p(noise, weights, coins, n_boot=800, seed=5)
    check("the wild test separates the signal panel from a noise panel",
          wild["p"] < w_noise["p"])
    check("the sign test rejects a unanimous count",
          sign_test_p(len(nets), len(nets)) < 1e-10)
    check("the sign test grades a split by how lopsided it is",
          sign_test_p(len(nets) // 2, len(nets)) == 1.0
          > sign_test_p(int(len(nets) * 0.6), len(nets))
          > sign_test_p(int(len(nets) * 0.875), len(nets)))


def part_three():
    print("\n3. The two conditions a one-way interval cannot see")
    panel = make_panel(n_coins=8, n_dates=10, n_events=3000, seed=4,
                       informed_low=0.88, informed_high=0.96,
                       date_shock_sd=0.8, weight_skew=1.2)
    nets, weights, coins, dates = [], [], [], []
    for row in panel:
        d = decompose(row["fill_price"], row["mid_at_fill"],
                      row["mid_at_horizon"], row["side"], row["size"])
        nets.append(d["net_bp"])
        weights.append(d["n_fills"])
        coins.append(row["coin"])
        dates.append(row["date"])
    nets = np.asarray(nets)
    one = cluster_bootstrap(nets, weights, coins, n_boot=800, seed=5)
    two = cluster_bootstrap(nets, weights, coins, n_boot=800, seed=5,
                            cluster_b=dates)
    ratio = max(weights) / min(weights)
    print(f"   fill-count spread across cells {ratio:.0f}x")
    print(f"   raw clusters {one['n_clusters']}, effective "
          f"{one['n_effective_clusters']:.2f}, t df {one['t_degrees_of_freedom']}")
    print(f"   one-way SE {one['se_cluster']:.4f}, verdict SE with dates "
          f"{two['se_used_for_verdict']:.4f}")
    check("the panel really is weight-skewed, so the effective count bites",
          ratio > 10 and one["n_effective_clusters"] < one["n_clusters"] - 0.5)
    half = (one["ci95_t"][1] - one["ci95_t"][0]) / 2.0
    known = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
             7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
    df_used = max(1, int(one["n_effective_clusters"])) - 1
    check("the t interval uses a published critical value, not the module's",
          df_used in known
          and abs(half - known[df_used] * one["se_cluster"]) < 1e-9)
    check("adding the date dimension widens the verdict interval here",
          two["se_used_for_verdict"] > one["se_cluster"] * 1.05)
    check("the two-way verdict is flagged as such",
          two["verdict_is_two_way"] and not one["verdict_is_two_way"])

    floor_ok = True
    for s in range(200):
        rng = np.random.default_rng(s)
        v = rng.standard_normal(50)
        w = np.exp(rng.standard_normal(50) * 1.5)
        c = [f"c{i % 5}" for i in range(50)]
        r = wild_cluster_p(v, w, c)
        if r["p"] < r["smallest_attainable_p"] - 1e-12:
            floor_ok = False
            break
    check("the wild p never falls below its floor over 200 panels", floor_ok)
    obs = wild_cluster_p(np.abs(nets) + 5.0, weights, coins)
    check("an unambiguous panel returns the floor, not zero",
          obs["p"] > 0 and abs(obs["p"] - obs["smallest_attainable_p"]) < 1e-12)

    rng_small = np.random.default_rng(17)
    conc_c = [f"c{i // 10}" for i in range(80)]
    conc_v = np.array([-0.9 if i < 10 else -0.7 for i in range(80)])
    conc_w = np.array([1e5 if i < 10 else 1.0 for i in range(80)])
    conc = cluster_bootstrap(conc_v, conc_w, conc_c, n_boot=400, seed=2)
    print(f"   weight-concentrated panel: {conc['n_clusters']} raw clusters, "
          f"{conc['n_effective_clusters']:.2f} effective")
    check("a panel below the effective-cluster floor returns no verdict",
          conc["n_clusters"] >= MIN_CLUSTERS
          and conc["n_effective_clusters"] < MIN_CLUSTERS
          and not conc["clears_zero"] and not conc["clears_zero_percentile"])
    check("abstaining is distinguishable from testing and not clearing",
          not conc["verdict_is_available"] and one["verdict_is_available"])
    check("the two-way flag reflects use, not merely being asked",
          two["verdict_is_two_way"] is (two["se_used_for_verdict"] > one["se_cluster"]))

    few_dates = [f"d{i % 4}" for i in range(len(nets))]
    fd = cluster_bootstrap(nets, weights, coins, n_boot=400, seed=5,
                           cluster_b=few_dates)
    check("few date clusters pull the degrees of freedom down",
          not fd["verdict_is_two_way"]
          or fd["t_degrees_of_freedom"] <= 3)

    flat = np.concatenate([np.full(40, -1.0), np.full(40, 1.0)])
    flat_c = [f"c{i % 8}" for i in range(80)]
    flat_d = [f"d{i // 8}" for i in range(80)]
    fl = cluster_bootstrap(flat, np.ones(80), flat_c, n_boot=400, seed=5,
                           cluster_b=flat_d)
    check("a two-way SE that does not widen is not labelled two-way",
          fl["verdict_is_two_way"]
          == (fl["se_used_for_verdict"] > fl["se_cluster"]))

    strong_c = [f"c{i % 6}" for i in range(180)]
    offs = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    strong_v = np.array([offs[i % 6] + rng_small.standard_normal() * 0.05
                         for i in range(180)])
    sb = cluster_bootstrap(strong_v, np.ones(180), strong_c, n_boot=600, seed=3)
    naive = float(strong_v.std(ddof=1) / np.sqrt(180))
    width = sb["ci95"][1] - sb["ci95"][0]
    check("the percentile interval resamples clusters, not observations",
          width > 4 * (2 * 1.96 * naive))
    contrib = np.array([np.sum(strong_v[[i for i, c in enumerate(strong_c)
                                         if c == u]] - sb["point"]) / 180
                        for u in sorted(set(strong_c))])
    g = len(contrib)
    check("the cluster standard error carries its finite-sample correction",
          abs(sb["se_cluster"]
              - float(np.sqrt((contrib ** 2).sum() * g / (g - 1)))) < 1e-9)
    sparse_rng = np.random.default_rng(SPARSE_SEED)
    sparse_v = sparse_rng.standard_normal(60)
    sparse_a = [f"a{i % 5}" for i in range(60)]
    sparse_b = [f"b{i // 5 % 7}" for i in range(60)]
    sparse_tw = two_way_se(sparse_v, np.ones(60), sparse_a, sparse_b)
    sparse = cluster_bootstrap(sparse_v, np.ones(60), sparse_a, n_boot=400,
                               seed=9, cluster_b=sparse_b)
    check("the chosen panel really does defeat the two-way variance",
          not np.isfinite(sparse_tw["se_two_way"]))
    check("a non-positive two-way variance falls back to the wider one-way",
          sparse["se_used_for_verdict"]
          >= max(sparse_tw["se_a"], sparse_tw["se_b"]) - 1e-12)

    check("the t critical value widens at small degrees of freedom",
          t_crit_95(4) > 2.7 and t_crit_95(4) > t_crit_95(30) > 1.96)
    flat = np.full(600, 1.0)
    sampled = wild_cluster_p(flat, np.ones(600),
                             [f"c{i % 30}" for i in range(600)], n_boot=400)
    check("the sampled wild branch cannot return exactly zero",
          not sampled["is_exact_enumeration"] and sampled["p"] > 0)

    many = wild_cluster_p(np.full(1100, 1.0), np.ones(1100),
                          [f"c{i}" for i in range(1100)], n_boot=200)
    check("the wild test survives a cluster count past the enumeration limit",
          not many["is_exact_enumeration"] and np.isfinite(many["p"]))

    empty = cluster_bootstrap([], [], [])
    check("the empty-panel result carries the same keys as a real one",
          set(empty) == set(one))


def part_four():
    print("\nPart four: the posting and fill rule the panels rest on")
    snap, bpx, bsz, apx, asz = [0], [100.0], [5.0], [101.0], [5.0]

    def run(ts, px, sz, side, s=snap, b=bsz):
        return simulate_touch_fills(s, bpx * len(s), b, apx * len(s),
                                    [5.0] * len(s), ts, px, sz, side)[3]

    check("a trade smaller than the queue ahead does not fill the order",
          run([1], [100.0], [3.0], [-1]).size == 0)
    check("a trade larger than the queue fills only the spill above it",
          abs(float(run([1], [100.0], [5.6], [-1])[0]) - 0.6) < 1e-12)
    check("the fill is capped at the size still resting",
          abs(float(run([1], [100.0], [99.0], [-1])[0]) - 1.0) < 1e-12)
    check("an aggressor on our own side does not fill us",
          run([1], [100.0], [7.0], [1]).size == 0)
    check("a trade away from the quote price does not fill",
          run([1], [100.5], [7.0], [-1]).size == 0)
    check("a trade before the posting instant cannot fill",
          simulate_touch_fills([10], bpx, bsz, apx, asz,
                               [1], [100.0], [7.0], [-1])[3].size == 0)
    check("an exactly consumed order emits one row and not a dust row",
          run([1, 2], [100.0, 100.0], [6.0, 6.0], [-1, -1]).size == 1)
    two = simulate_touch_fills([0, 10], [100.0, 100.0], [5.0, 5.0],
                               [101.0, 101.0], [5.0, 5.0],
                               [1, 11], [100.0, 100.0], [7.0, 7.0], [-1, -1])[3]
    check("queue priority does not carry across snapshots",
          two.size == 2 and abs(float(two.sum()) - 2.0) < 1e-12)
    check("no quote is posted when one side of the book is missing",
          simulate_touch_fills([0], [100.0], [5.0], [float("nan")], [0.0],
                               [1], [100.0], [7.0], [-1])[3].size == 0)


def part_five():
    """The t tail and the two multiplicity procedures, against known values.

    Both are implemented in this package rather than pulled from scipy, which
    keeps the single numpy dependency. Reference values below were taken from
    scipy and are hardcoded so the check does not need it.
    """
    print("\nPart five: the t tail and the multiplicity procedures")
    for df, t, want in ((4, 2.776, 0.05002278), (8, 2.306, 0.05000032),
                        (29, 2.045, 0.05002408), (1, 12.706, 0.05000080),
                        (200, 1.960, 0.05138484)):
        check(f"two-sided t p at df {df} matches the reference to 1e-7",
              abs(t_two_sided_p(t, df) - want) < 1e-7)
    check("a zero t statistic returns p of one",
          abs(t_two_sided_p(0.0, 10) - 1.0) < 1e-12)
    check("a t p-value is refused below one degree of freedom",
          t_two_sided_p(2.0, 0) != t_two_sided_p(2.0, 0))

    raw = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.5]
    want_bh = [0.007, 0.028, 0.0588, 0.0588, 0.0588, 0.07, 0.5]
    got = benjamini_hochberg(raw)
    check("Benjamini-Hochberg matches the reference on a known family",
          all(abs(a - b) < 1e-9 for a, b in zip(got, want_bh)))
    check("Benjamini-Hochberg is monotone in the sorted p-values",
          all(x <= y + 1e-12 for x, y in zip(sorted(got), sorted(got)[1:])))
    check("a NaN p-value stays NaN and leaves the family",
          np.isnan(benjamini_hochberg([0.01, float("nan")])[1])
          and abs(benjamini_hochberg([0.01, float("nan")])[0] - 0.01) < 1e-12)
    check("an adjusted p-value is never below its raw value",
          all(a >= b - 1e-12 for a, b in zip(got, raw)))


def main():
    print("Synthetic smoke test for the entry-markout decomposition\n")
    part_one()
    part_one_identities()
    part_two()
    part_three()
    part_four()
    part_five()
    failed = [n for n, ok in CHECKS if not ok]
    print(f"\n{len(CHECKS) - len(failed)} of {len(CHECKS)} checks passed.")
    if failed:
        for n in failed:
            print(f"  FAILED: {n}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
