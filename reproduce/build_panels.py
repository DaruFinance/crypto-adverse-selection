"""Rebuild the nine shipped panels from frozen measurement artifacts.

The inputs are derived measurement outputs, not raw order-book archives. Their
relative paths and hashes are recorded in ``lineage.json``. This script keeps
the boundary explicit while making the final extraction into the published
coin-day panels reproducible.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "lineage.json"

CORE_COLUMNS = [
    "coin", "date", "n_fills", "mean_fill_size",
    "spread_capture_bp_10s", "adverse_select_bp_10s",
    "net_markout_bp_10s", "spread_capture_bp_60s",
    "adverse_select_bp_60s", "net_markout_bp_60s",
    "spread_capture_szw_bp_10s", "adverse_select_szw_bp_10s",
    "net_markout_szw_bp_10s", "spread_capture_szw_bp_60s",
    "adverse_select_szw_bp_60s", "net_markout_szw_bp_60s", "rv_10s_bp",
]

ARM_COLUMNS = [
    "coin", "date", "touch_net_bp", "touch_fills", "as_net_bp", "as_fills",
    "diff_bp", "n_opportunities", "n_events", "as_frac_quotes_outside_book",
    "as_frac_quotes_at_deepest_level", "as_mean_abs_inventory", "k_per_bp",
    "floor_half_spread_bp",
]

LEADLAG_COLUMNS = [
    "coin", "date", "matched_peak_lag_ms", "matched_peak_corr",
    "raw_peak_lag_ms", "raw_peak_corr", "bybit_cadence_ms", "hl_cadence_ms",
]

CONDITIONAL_COLUMNS = [
    "coin", "date", "block", "regime", "capture_sum", "adverse_sum",
    "net_sum", "n_fills",
]

REQUOTE_COLUMNS = [
    "venue", "coin", "date", "gap_ms", "n_fills", "capture_bp",
    "adverse_bp", "net_bp",
]

SIGN_RULE_COLUMNS = [
    "venue", "coin", "date", "rule", "n_fills",
    "spread_capture_bp_10s", "adverse_select_bp_10s", "net_markout_bp_10s",
    "spread_capture_bp_60s", "adverse_select_bp_60s", "net_markout_bp_60s",
    "n_trades", "sign_accuracy", "sign_accuracy_szw",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_paths(root: Path, paths: list[str]) -> str:
    digest = hashlib.sha256()
    for rel in sorted(paths):
        path = root / rel
        digest.update(rel.replace("\\", "/").encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def sha256_depth_tree(root: Path, rel: str) -> str:
    base = root / rel
    files = sorted(base.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no JSON files under {base}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def write_rows(path: Path, columns: list[str], rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for source in rows:
            writer.writerow({column: source.get(column, "") for column in columns})
            count += 1
    return count


def core_rows(path: Path, coin_key: str):
    with path.open(newline="") as handle:
        for source in csv.DictReader(handle):
            coin = source.get(coin_key, "")
            date = source.get("date", "").replace("-", "")
            if not coin or not date:
                continue
            row = {column: source.get(column, "") for column in CORE_COLUMNS}
            row["coin"] = coin
            row["date"] = date
            yield row


def depth_rows(path: Path):
    for source_path in sorted(path.glob("*.json")):
        source = json.loads(source_path.read_text())
        block = source["M7_depth_rank_economics"]
        markouts = block["mean_markout_by_rank_bp"]
        fills = block["n_fills_by_rank"]
        opportunities = block["n_opportunities_by_rank"]
        for index, (markout, n_fills, n_opportunities) in enumerate(
                zip(markouts, fills, opportunities), start=1):
            yield {
                "coin": source["root"], "date": str(source["date"]),
                "level": index, "mean_markout_bp": markout,
                "n_fills": n_fills, "n_opportunities": n_opportunities,
            }


def jsonl_rows(path: Path):
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if "_meta" not in row:
                yield row


def json_rows(path: Path, key: str):
    yield from json.loads(path.read_text())[key]


def requote_rows(path: Path, venue: str):
    for source in json_rows(path, "per_coinday"):
        yield {
            "venue": venue, "coin": source["coin"], "date": source["date"],
            "gap_ms": source["gap_ms"], "n_fills": source["n_fills"],
            "capture_bp": source["capture_bp"],
            "adverse_bp": source["adverse_bp"], "net_bp": source["net_bp"],
        }


def sign_rule_rows(root: Path, out: Path):
    """The sign-rule counterfactual, restricted to the coin-days each venue's
    own panel carries.

    The measurement program sweeps every coin-day on disk, while the two venue
    panels are post-dust-guard subsets of that. Restricting here is what makes
    the ``true`` rows reconcile with the headline of the main decomposition
    instead of sitting on a slightly wider sample; the restriction is read off
    the two panels built earlier in this same run, so it is fixed by the
    manifest rather than by a list kept in this file.
    """
    sources = [
        ("bybit_perp", "runs/sign_rule/bybit_sign_rules.csv",
         "bybit_perp_coindays.csv"),
        ("hyperliquid", "runs/sign_rule/hl_sign_rules.csv",
         "hyperliquid_coindays.csv"),
    ]
    for venue, rel, panel_name in sources:
        with (out / panel_name).open(newline="") as handle:
            keep = {(r["coin"], r["date"]) for r in csv.DictReader(handle)}
        with (root / rel).open(newline="") as handle:
            for source in csv.DictReader(handle):
                if (source["coin"], source["date"]) not in keep:
                    continue
                row = {c: source.get(c, "") for c in SIGN_RULE_COLUMNS}
                row["venue"] = venue
                yield row


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def verify_outputs(root: Path, manifest: dict) -> bool:
    failed = False
    for name, spec in manifest["panels"].items():
        path = root / name
        if not path.exists():
            print(f"FAIL  missing {name}")
            failed = True
            continue
        actual = sha256_file(path)
        expected = spec["output_sha256"]
        label = "ok  " if actual == expected else "FAIL"
        print(f"{label}  {name}  {actual}")
        failed |= actual != expected
    return not failed


def verify_sources(root: Path, manifest: dict) -> bool:
    failed = False
    for name, spec in manifest["panels"].items():
        paths = spec["source_paths"]
        if spec["source_kind"] == "json_directory":
            actual = sha256_depth_tree(root, paths[0])
        elif len(paths) == 1:
            actual = sha256_file(root / paths[0])
        else:
            actual = sha256_paths(root, paths)
        expected = spec["source_sha256"]
        label = "ok  " if actual == expected else "FAIL"
        print(f"{label}  source for {name}  {actual}")
        failed |= actual != expected
    return not failed


def build(source_root: Path, out: Path) -> None:
    jobs = [
        ("bybit_perp_coindays.csv", CORE_COLUMNS,
         core_rows(source_root / "runs/bybit/decomp_nodust.csv", "coin")),
        ("binance_um_coindays.csv", CORE_COLUMNS,
         core_rows(source_root / "runs/stream/decomp_nodust_full15.csv", "root")),
        ("hyperliquid_coindays.csv", CORE_COLUMNS,
         core_rows(source_root / "runs/hl10_decomp_merged.csv", "coin")),
        ("bybit_depth_levels.csv",
         ["coin", "date", "level", "mean_markout_bp", "n_fills", "n_opportunities"],
         depth_rows(source_root / "runs/mm_depth3")),
        ("bybit_quoting_arm_coindays.csv", ARM_COLUMNS,
         jsonl_rows(source_root / "runs/frontier/as_quoter_arm_v5.json.rows.jsonl")),
        ("cross_venue_leadlag_coindays.csv", LEADLAG_COLUMNS,
         json_rows(source_root / "runs/part2/leadlag_sweep_250ms.json", "per_coinday")),
        ("bybit_conditional_cells.csv", CONDITIONAL_COLUMNS,
         core_rows_passthrough(
             source_root / "runs/conditional_bybit_v15_common_strided_per_coinday_cells.csv")),
        ("venue_requote_rungs.csv", REQUOTE_COLUMNS,
         combined_requote_rows(source_root)),
        ("sign_rule_coindays.csv", SIGN_RULE_COLUMNS,
         sign_rule_rows(source_root, out)),
    ]
    for name, columns, rows in jobs:
        count = write_rows(out / name, columns, rows)
        print(f"built  {name}  {count:,} rows")


def core_rows_passthrough(path: Path):
    with path.open(newline="") as handle:
        yield from csv.DictReader(handle)


def combined_requote_rows(root: Path):
    yield from requote_rows(root / "runs/part2/requote_bybit_25d.json", "bybit_perp")
    yield from requote_rows(root / "runs/part2/requote_hl_25d_all.json", "hyperliquid")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--out", type=Path, default=HERE / "panels")
    parser.add_argument("--verify-shipped", action="store_true")
    parser.add_argument("--skip-source-hash", action="store_true")
    args = parser.parse_args()
    manifest = load_manifest()

    if args.verify_shipped:
        raise SystemExit(0 if verify_outputs(HERE / "panels", manifest) else 1)
    if args.source_root is None:
        parser.error("--source-root is required unless --verify-shipped is used")
    if not args.skip_source_hash and not verify_sources(args.source_root, manifest):
        raise SystemExit("source hash mismatch; refusing to rebuild frozen panels")
    build(args.source_root, args.out)
    raise SystemExit(0 if verify_outputs(args.out, manifest) else 1)


if __name__ == "__main__":
    main()
