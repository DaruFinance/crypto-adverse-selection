"""
Print the headline numbers in the paper from the aggregated result files.

Run from the repository root: python reproduce/print_headline.py

Every figure printed here is read from a JSON file in this directory; nothing
is recomputed. The per-venue block prints the coin counts that qualify the
pooled result, including coins that run the other way, so the output cannot
show the pooled number without also showing what disagrees with it.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name):
    return json.loads((HERE / name).read_text())


def main():
    d = load("decomposition_by_venue.json")
    print("Entry-markout decomposition, 10s horizon, pre-fee, basis points")
    print(f"{'venue':14s} {'coin-days':>10} {'capture':>9} {'adverse':>9} "
          f"{'net':>9} {'adv/cap':>8} {'coins neg':>10} {'clear 0':>8}")
    for venue, v in d["venues"].items():
        p = v["pooled"]["10s_row_mean"]
        print(f"{venue:14s} {v['n_coindays']:>10} {p['capture_bp']:>9.4f} "
              f"{p['adverse_bp']:>9.4f} {p['net_bp']:>9.4f} "
              f"{p['adv_over_cap']:>8.3f} "
              f"{str(v['n_coins_net_negative']) + '/' + str(v['n_coins']):>10} "
              f"{v['n_coins_net_negative_clearing_zero']:>8}")

    print("\nWhat runs against the pooled sign")
    for venue, v in d["venues"].items():
        pos = v["coins_significantly_positive"]
        loo = v["leave_one_coin_out"]
        print(f"  {venue:14s} counterexamples to net <= 0: "
              f"{v['counterexamples_to_net_le_zero']}, "
              f"significantly positive: {', '.join(pos) if pos else 'none'}")
        print(f"  {'':14s} coins whose own net runs opposite to the pool: "
              f"{loo['n_coins_opposite_sign_to_pooled']} of {v['n_coins']}"
              f"{': ' + ', '.join(loo['coins_opposite_sign']) if loo['coins_opposite_sign'] else ''}")
        print(f"  {'':14s} pooled sign survives dropping any single coin: "
              f"{loo['sign_stable']}")

    vd = load("venue_difference.json")
    print("\nVenue difference, matched re-quote rung")
    print(f"  {vd['a']} minus {vd['b']} at {vd['gap_ms']} ms")
    print(f"  pooled {vd['pooled_diff_bp']:+.4f}  paired "
          f"{vd['paired_weighted_diff_bp']:+.4f}  "
          f"unweighted {vd['paired_unweighted_diff_bp']:+.4f}")
    print(f"  {vd['n_shared_coindays']} shared coin-days over "
          f"{vd['n_date_clusters']} date clusters")
    print("  descriptive and extraction-dependent, not a causal venue comparison")

    pc = load("per_coin_intervals.json")
    print("\nPer-coin intervals")
    for venue, v in pc["venues"].items():
        cells = [t for c in v["per_coin"].values() for t in c.values()
                 if isinstance(t, dict)]
        cleared = [t for t in cells if t.get("clears_zero")]
        powerless = sum(1 for t in cleared
                        if not t.get("percentile_interval_has_power_to_fail"))
        print(f"  {venue:14s} {len(cleared)} cells clear zero, and {powerless} of "
              f"those sit on an interval that could not have reached it")
    print(f"  {pc['n_testable_cells_total']} cells across two horizons; "
          f"{pc['n_cells_without_power_total']} sit on an interval that cannot "
          f"reach zero ({pc['frac_cells_without_power']:.0%})")

    dr = load("depth_rebate_frontier.json")
    print("\nDepth frontier")
    print(f"  markout by level {[round(x, 4) for x in dr['markout_bp_by_level']]}")
    print(f"  fill rate L5 over L1 {dr['fill_rate_ratio_L5_over_L1']:.4f}")
    print(f"  minimum rebate before any level profits "
          f"{dr['min_rebate_for_any_level_profitable_bp']:.4f} bp")
    print(f"  touch overtakes the deepest level at "
          f"{dr['touch_overtakes_deepest_at_rebate_bp']:.4f} bp against a "
          f"published tier of {dr['bybit_best_published_tier_bp']} bp")

    asm = load("avellaneda_stoikov_arm.json")
    print("\nAvellaneda-Stoikov quoting arm, Bybit only")
    print(f"  verdict {asm['verdict']}, net {asm['as_net_bp']:+.4f} bp, "
          f"negative on {asm['as_net_negative_on']}")
    print(f"  fill share of the touch {asm['as_fill_share_of_touch']:.4f}")

    print("\nConditional decomposition, net of adverse selection, Bybit only")
    for block in ("asia", "europe", "us"):
        c = load(f"conditional_net_{block}.json")
        print(f"  {block:7s} {c['n_coins_positive']}/{c['n_coins']} coins "
              f"positive, dependence-robust p {c['recommended_p']:.4f}")
    print("  No block has a detectable difference under this low-powered test.")
    print("  Failure to reject does not establish invariance. The gross")
    print("  legs are not reported, because the choice of weighting and of which")
    print("  cells to include moves them and was not fixed in advance.")

    ll = load("cross_venue_leadlag.json")
    m = ll["matched"]
    print("\nCross-venue lead-lag on the same coin")
    print(f"  median peak lag {m['median_peak_lag_ms']:.0f} ms, leads on "
          f"{m['n_bybit_leads']} of {ll['n_coindays']} coin-days")
    cad = ll["cadence"]
    print(f"  the venues publish {cad['gap_ms']:.0f} ms apart, so this cannot "
          f"separate a lead from a cadence difference")


if __name__ == "__main__":
    main()
