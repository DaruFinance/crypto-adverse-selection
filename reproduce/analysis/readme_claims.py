"""Check the README's load-bearing claims against shipped result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "reproduce"


def load(name: str):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def _signrule_row(prefix: str, block: dict, rule: str) -> str:
    """One row of the README's sign-rule table, normalized to single spaces."""
    r = block["by_rule"][rule]
    fields = [
        prefix,
        f"{r['sign_accuracy']:.3f}",
        f"{r['capture_bp_10s']:.3f}",
        f"{r['adverse_bp_10s']:.3f}",
        f"{r['net_bp_10s']:.3f}",
        f"{r['net_error_bp_10s']:+.3f}",
        f"{r['fill_count_ratio']:.3f}",
    ]
    return " ".join(" ".join(fields).split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    normalized_readme = " ".join(readme.split())
    decomp = load("decomposition_by_venue.json")["venues"]
    venue = load("venue_difference.json")
    leadlag = load("cross_venue_leadlag.json")
    depth = load("depth_rebate_frontier.json")
    arm = load("avellaneda_stoikov_arm.json")
    signrule = load("sign_rule_counterfactual.json")["venues"]
    multiplicity = load("per_coin_multiplicity.json")
    conditional = {
        block: load(f"conditional_net_{block}.json")
        for block in ("asia", "europe", "us")
    }
    contract = load("claim_contract.json")

    venue_rows = {
        "bybit_perp": "Bybit perpetuals 0.206 -1.044 -0.839 [-1.084, -0.594]",
        "binance_um": "Binance USD-M perpetuals 0.460 -0.852 -0.391 [-0.452, -0.331]",
        "hyperliquid": "Hyperliquid 0.560 -1.017 -0.458 [-0.729, -0.186]",
    }
    expected = {}
    for key, template in venue_rows.items():
        row = decomp[key]["pooled"]["10s_row_mean"]
        lo, hi = row["net_ci95_t_two_way_month_coin"]
        expected[f"headline_{key}"] = template.format(
            capture=row["capture_bp"],
            adverse=row["adverse_bp"],
            net=row["net_bp"],
            lo=lo,
            hi=hi,
        )

    expected.update(
        {
            "venue_difference": (
                f"Bybit-minus-Hyperliquid estimate is {venue['pooled_diff_bp']:.3f} bp"
            ),
            "leadlag": (
                f"median {leadlag['matched']['median_peak_lag_ms']:.0f} ms Bybit lead"
            ),
            "cadence": (
                f"publication-cadence gap is {leadlag['cadence']['gap_ms']:.0f} ms"
            ),
            "conditional": "p-values of "
            + f"{conditional['asia']['recommended_p']:.3f}, "
            + f"{conditional['europe']['recommended_p']:.3f} and "
            + f"{conditional['us']['recommended_p']:.3f}",
            "depth_scope": f"{depth['n_coindays']}-coin-day, bid-side Bybit depth panel",
            "depth_min_rebate": (
                f"break-even rebate is {depth['min_rebate_for_any_level_profitable_bp']:.3f} bp"
            ),
            "depth_crossing": (
                f"deepest quoted level at {depth['touch_overtakes_deepest_at_rebate_bp']:.3f} bp"
            ),
            "arm_negative": "negative on "
            + arm["as_net_negative_on"].replace("/", " of ")
            + " coin-days",
            "arm_fill_share": f"takes {100 * arm['as_fill_share_of_touch']:.1f}% as many fills",
            "lineage_boundary": "The reproducible chain starts at the frozen artifacts and ends at the figures; raw-archive processing remains outside it",
            "markout_boundary": "does not establish realised maker profit or loss",
            "signrule_bybit_lee_ready": _signrule_row(
                "Bybit        Lee-Ready    ", signrule["bybit_perp"], "lee_ready"),
            "signrule_bybit_tick": _signrule_row(
                "Bybit        tick rule    ", signrule["bybit_perp"], "tick"),
            "signrule_binance_lee_ready": _signrule_row(
                "Binance      Lee-Ready    ", signrule["binance_um"], "lee_ready"),
            "signrule_binance_tick": _signrule_row(
                "Binance      tick rule    ", signrule["binance_um"], "tick"),
            "signrule_hl_lee_ready": _signrule_row(
                "Hyperliquid  Lee-Ready    ", signrule["hyperliquid"], "lee_ready"),
            "signrule_hl_tick": _signrule_row(
                "Hyperliquid  tick rule    ", signrule["hyperliquid"], "tick"),
            "multiplicity_bh": (
                f"{multiplicity['counts']['raw_below_alpha']} clear zero at raw p below 0.05 "
                f"and all {multiplicity['counts']['bh_below_alpha_both_horizons']} survive "
                f"Benjamini-Hochberg; "
                f"{multiplicity['counts']['by_below_alpha_both_horizons']} survive "
                f"Benjamini-Yekutieli"),
            "multiplicity_family": (
                f"the {multiplicity['n_cells_in_family']} per-coin cells that return a verdict"),
        }
    )

    checks = []
    for name, text in expected.items():
        passed = text in normalized_readme
        checks.append({"name": name, "expected_text": text, "passed": passed})

    folded = readme.casefold()
    for phrase in contract["prohibited"]:
        passed = phrase.casefold() not in folded
        checks.append(
            {"name": f"prohibited:{phrase}", "expected_absent": phrase, "passed": passed}
        )

    report = {
        "n_checks": len(checks),
        "n_passed": sum(item["passed"] for item in checks),
        "checks": checks,
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    failed = [item for item in checks if not item["passed"]]
    if failed:
        raise SystemExit("README claim check failed: " + ", ".join(x["name"] for x in failed))
    print(f"README claim check passed: {report['n_passed']}/{report['n_checks']}")


if __name__ == "__main__":
    main()
