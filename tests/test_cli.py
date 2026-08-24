"""
Tests for ``makercex measure``.

This is the module a stranger runs first and almost all of it is input
handling, which is where the silent failures live. A wrong timestamp unit, an
aggressor column read backwards or a column matched to the wrong header do not
raise: they return a plausible basis-point figure computed from the wrong
thing. So the checks below are mostly about the boundary rather than the
arithmetic, and each one names the failure it would catch.

The fixture is a deterministic synthetic book and tape, generated here rather
than shipped, so the tests carry no data and depend on no archive. It is built
so the reference quoter takes a known, non-trivial number of fills: the tape
alternates aggressor side, trades at the quoted price and carries size above
the resting queue, which is the only way the last-in-queue rule fills at all.

Run under pytest, or directly:

    python -m pytest tests/
    python tests/test_cli.py
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from makercex import MIN_CLUSTERS
from makercex.cli import (Bail, _to_ns, _to_side, build_parser, load_inputs,
                          measure, parse_horizon)

DAY_NS = 86_400_000_000_000
EPOCH = 1_700_000_000_000_000_000     # 2023-11-14, a real calendar instant


def make_book_and_tape(n_days=12, n_snaps=40, symbols=("AAA", "BBB", "CCC"),
                       start_ns=EPOCH, day_stride_ns=31 * DAY_NS):
    """A deterministic book and tape as two lists of row dicts.

    ``day_stride_ns`` spaces the days a month apart by default, so the panel
    carries one cluster per day and the number of month clusters is under the
    caller's control. That is what lets a test drive the abstention rule from
    either side of its floor.
    """
    rng = np.random.default_rng(11)
    snaps, trades = [], []
    for day in range(n_days):
        day0 = start_ns + day * day_stride_ns
        for sym in symbols:
            mid = 100.0 + 5.0 * symbols.index(sym)
            for k in range(n_snaps):
                ts = day0 + k * 1_000_000_000
                mid *= float(np.exp(rng.standard_normal() * 2e-4))
                bid, ask = round(mid - 0.05, 4), round(mid + 0.05, 4)
                snaps.append({"ts_ns": ts, "bid_px": bid, "bid_sz": 5.0,
                              "ask_px": ask, "ask_sz": 5.0, "symbol": sym})
                # One trade inside the interval, alternating side, priced at the
                # quote and larger than the queue ahead so the spill fills us.
                aggressor_buys = (k % 2 == 0)
                trades.append({
                    "ts_ns": ts + 100_000_000,
                    "price": ask if aggressor_buys else bid,
                    "size": 6.5,
                    "side": 1 if aggressor_buys else -1,
                    "symbol": sym,
                })
    return snaps, trades


def write_csv(path, rows):
    with Path(path).open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return str(path)


def write_parquet(path, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq
    cols = {k: [r[k] for r in rows] for k in rows[0]}
    pq.write_table(pa.table(cols), str(path))
    return str(path)


def run(snapshots, trades, **overrides):
    """Parse args the way the CLI does, then measure."""
    argv = ["measure", "--snapshots", str(snapshots), "--trades", str(trades)]
    for k, v in overrides.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    args = build_parser().parse_args(argv)
    return measure(load_inputs(args), parse_horizon(args.horizon),
                   size=args.size, n_boot=args.n_boot, seed=args.seed)


# --------------------------------------------------------------------------
# horizons


def test_horizon_units_parse_to_nanoseconds():
    assert parse_horizon("10s") == 10_000_000_000
    assert parse_horizon("60s") == 60_000_000_000
    assert parse_horizon("500ms") == 500_000_000
    assert parse_horizon("2m") == 120_000_000_000
    assert parse_horizon("1h") == 3_600_000_000_000


def test_a_horizon_that_cannot_be_read_is_refused():
    for bad in ("ten seconds", "10", "s", "", "10x"):
        try:
            parse_horizon(bad)
        except Bail:
            continue
        raise AssertionError(f"{bad!r} should not parse as a horizon")


# --------------------------------------------------------------------------
# the aggressor flag, which is the one input that cannot be inferred


def test_aggressor_side_spellings_agree():
    for enc in (["1", "-1"], ["buy", "sell"], ["b", "s"], ["BUY", "SELL"],
                ["bid", "ask"], [1, -1]):
        assert list(_to_side(enc, "side")) == [1, -1], enc


def test_a_maker_flag_is_inverted_in_every_encoding():
    """is_buyer_maker true means the buyer rested, so the aggressor sold.

    The integer encoding is the one that matters: the venues publishing this
    field ship 0/1 in their CSV archives, and reading a 1 there as a buy flips
    the sign on every trade in the file, turning captured spread into adverse
    selection without raising anything.
    """
    for enc in ([True, False], ["true", "false"], ["TRUE", "FALSE"],
                ["1", "0"], [1, 0]):
        assert list(_to_side(enc, "is_buyer_maker")) == [-1, 1], enc


def test_an_unreadable_side_is_refused_rather_than_guessed():
    for enc, col in ((["up", "down"], "side"), (["maybe"], "is_buyer_maker")):
        try:
            _to_side(enc, col)
        except Bail:
            continue
        raise AssertionError(f"{enc!r} in {col!r} should not be readable")


# --------------------------------------------------------------------------
# timestamps


def test_every_unit_infers_to_the_same_instant():
    """The same instant in s, ms, us and ns must land on the same nanosecond.

    A missed unit does not fail, it rescales every horizon in the run, so this
    is the check that the markout window means what it says.
    """
    base_s = 1_700_000_000
    want = base_s * 1_000_000_000
    for mult in (1, 10 ** 3, 10 ** 6, 10 ** 9):
        got = _to_ns([base_s * mult, base_s * mult + mult])
        assert int(got[0]) == want, mult


def test_an_implausible_inferred_unit_is_refused():
    """No unit puts these values in a sane calendar, so the run stops."""
    try:
        _to_ns([1_700_000, 1_700_001])
    except Bail as exc:
        assert "--time-unit" in str(exc)
        return
    raise AssertionError("an implausible inferred unit should be refused")


def test_an_explicit_unit_overrides_the_calendar_check():
    assert int(_to_ns([1_700_000], unit="s")[0]) == 1_700_000 * 10 ** 9


def test_iso_datetime_strings_are_read():
    when = datetime(2023, 11, 14, tzinfo=timezone.utc)
    got = _to_ns([when.isoformat(), (when + timedelta(seconds=1)).isoformat()])
    assert int(got[1] - got[0]) == 1_000_000_000


# --------------------------------------------------------------------------
# column resolution


def test_known_column_names_are_found(tmp_path):
    snaps, trades = make_book_and_tape()
    res = run(write_csv(tmp_path / "b.csv", snaps),
              write_csv(tmp_path / "t.csv", trades))
    assert res["n_fills"] > 0
    assert res["n_symbols"] == 3
    assert res["n_cells"] == 36


def test_odd_column_names_resolve_through_the_alias_table(tmp_path):
    """Same data under different spellings must give the identical result."""
    snaps, trades = make_book_and_tape()
    plain = run(write_csv(tmp_path / "b.csv", snaps),
                write_csv(tmp_path / "t.csv", trades))

    rename_s = {"ts_ns": "timestamp", "bid_px": "best_bid_price",
                "bid_sz": "bq", "ask_px": "best_ask_price", "ask_sz": "aq",
                "symbol": "instrument"}
    rename_t = {"ts_ns": "transact_time", "price": "px", "size": "quantity",
                "side": "aggressor_side", "symbol": "instrument"}
    odd = run(write_csv(tmp_path / "b2.csv",
                        [{rename_s[k]: v for k, v in r.items()} for r in snaps]),
              write_csv(tmp_path / "t2.csv",
                        [{rename_t[k]: v for k, v in r.items()} for r in trades]))
    assert odd["pooled"] == plain["pooled"]
    assert odd["n_fills"] == plain["n_fills"]


def test_a_column_outside_the_alias_table_needs_an_override(tmp_path):
    snaps, trades = make_book_and_tape()
    weird = [{("wallclock" if k == "ts_ns" else k): v for k, v in r.items()}
             for r in snaps]
    book = write_csv(tmp_path / "b.csv", weird)
    tape = write_csv(tmp_path / "t.csv", trades)
    try:
        run(book, tape)
    except Bail as exc:
        assert "--snapshot-time-col" in str(exc), exc
    else:
        raise AssertionError("an unknown timestamp column should be refused")
    assert run(book, tape, snapshot_time_col="wallclock")["n_fills"] > 0


def test_a_missing_column_names_a_flag_that_exists(tmp_path):
    """The failure message has to point at a real flag, not an invented one."""
    snaps, trades = make_book_and_tape()
    dropped = [{k: v for k, v in r.items() if k != "side"} for r in trades]
    parser_flags = set(build_parser().parse_args(
        ["measure", "--snapshots", "x", "--trades", "y"]).__dict__)
    try:
        run(write_csv(tmp_path / "b.csv", snaps),
            write_csv(tmp_path / "t.csv", dropped))
    except Bail as exc:
        flag = str(exc).rsplit("--", 1)[-1].strip().rstrip(".")
        assert flag.replace("-", "_") in parser_flags, exc
        return
    raise AssertionError("a missing side column should be refused")


# --------------------------------------------------------------------------
# file formats


def test_parquet_and_delimited_text_agree(tmp_path):
    snaps, trades = make_book_and_tape()
    as_csv = run(write_csv(tmp_path / "b.csv", snaps),
                 write_csv(tmp_path / "t.csv", trades))
    as_pq = run(write_parquet(tmp_path / "b.parquet", snaps),
                write_parquet(tmp_path / "t.parquet", trades))
    assert as_pq["n_fills"] == as_csv["n_fills"]
    assert as_pq["pooled"] == as_csv["pooled"]


def test_a_tab_separated_file_is_read(tmp_path):
    snaps, trades = make_book_and_tape()
    ref = run(write_csv(tmp_path / "b.csv", snaps),
              write_csv(tmp_path / "t.csv", trades))
    for name, rows in (("b.tsv", snaps), ("t.tsv", trades)):
        with (tmp_path / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
            w.writeheader(); w.writerows(rows)
    got = run(tmp_path / "b.tsv", tmp_path / "t.tsv")
    assert got["n_fills"] == ref["n_fills"]


# --------------------------------------------------------------------------
# the decomposition and its interval


def test_capture_is_fixed_at_fill_time_across_horizons(tmp_path):
    """Proposition 1(ii): only the adverse leg may move with the horizon."""
    snaps, trades = make_book_and_tape()
    book = write_csv(tmp_path / "b.csv", snaps)
    tape = write_csv(tmp_path / "t.csv", trades)
    a = run(book, tape, horizon="10s")
    b = run(book, tape, horizon="60s")
    assert abs(a["pooled"]["capture_bp"] - b["pooled"]["capture_bp"]) < 1e-12
    assert a["pooled"]["adverse_bp"] != b["pooled"]["adverse_bp"]


def test_net_is_capture_plus_adverse_and_the_rebate_is_its_negative(tmp_path):
    snaps, trades = make_book_and_tape()
    res = run(write_csv(tmp_path / "b.csv", snaps),
              write_csv(tmp_path / "t.csv", trades))
    p = res["pooled"]
    assert abs(p["capture_bp"] + p["adverse_bp"] - p["net_bp"]) < 1e-9
    assert abs(res["breakeven_rebate_bp"] + p["net_bp"]) < 1e-12


def test_the_abstention_rule_fires_below_the_floor(tmp_path):
    """Under five effective clusters the tool must return no verdict.

    A withheld verdict is not a failed test, and the distinction is the whole
    point of the rule, so both sides of the floor are checked.
    """
    thin, thin_t = make_book_and_tape(n_days=3)
    res = run(write_csv(tmp_path / "b1.csv", thin),
              write_csv(tmp_path / "t1.csv", thin_t))
    assert res["interval"]["n_effective_clusters"] < MIN_CLUSTERS
    assert res["interval"]["verdict_is_available"] is False
    assert res["interval"]["clears_zero"] is False

    wide, wide_t = make_book_and_tape(n_days=14)
    res = run(write_csv(tmp_path / "b2.csv", wide),
              write_csv(tmp_path / "t2.csv", wide_t))
    assert res["interval"]["n_effective_clusters"] >= MIN_CLUSTERS
    assert res["interval"]["verdict_is_available"] is True
    lo, hi = res["interval"]["ci95_t_two_way"] or res["interval"]["ci95_t"]
    assert lo < hi


def test_the_interval_crosses_two_dimensions_when_a_symbol_is_present(tmp_path):
    snaps, trades = make_book_and_tape(n_days=14)
    with_sym = run(write_csv(tmp_path / "b.csv", snaps),
                   write_csv(tmp_path / "t.csv", trades))
    assert with_sym["clustering"] == "month and symbol"

    strip = lambda rows: [{k: v for k, v in r.items() if k != "symbol"}
                          for r in rows]
    without = run(write_csv(tmp_path / "b2.csv", strip(snaps)),
                  write_csv(tmp_path / "t2.csv", strip(trades)))
    assert without["clustering"] == "month"


def test_a_tape_that_cannot_fill_says_why(tmp_path):
    """The queue rule fills only on the spill, so an undersized trade fills
    nothing. That is a configuration mistake, not an empty result."""
    snaps, trades = make_book_and_tape()
    small = [{**r, "size": 0.5} for r in trades]     # under the resting queue
    try:
        run(write_csv(tmp_path / "b.csv", snaps),
            write_csv(tmp_path / "t.csv", small))
    except Bail as exc:
        assert "no fills" in str(exc)
        return
    raise AssertionError("a tape that cannot fill should be refused")


# --------------------------------------------------------------------------
# the command as a stranger invokes it


def test_the_installed_command_runs_end_to_end(tmp_path):
    snaps, trades = make_book_and_tape(n_days=14)
    out = tmp_path / "result.json"
    proc = subprocess.run(
        [sys.executable, "-m", "makercex.cli", "measure",
         "--snapshots", write_csv(tmp_path / "b.csv", snaps),
         "--trades", write_csv(tmp_path / "t.csv", trades),
         "--horizon", "10s", "--json", str(out)],
        capture_output=True, text=True,
        cwd=str(Path(__file__).resolve().parents[1]))
    assert proc.returncode == 0, proc.stderr
    assert "net entry markout" in proc.stdout
    assert "not realised profit" in proc.stdout
    saved = json.loads(out.read_text())
    assert saved["n_cells"] == len(saved["cells"])


def test_a_bad_invocation_exits_nonzero_without_a_traceback(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "makercex.cli", "measure",
         "--snapshots", str(tmp_path / "nope.csv"),
         "--trades", str(tmp_path / "nope.csv")],
        capture_output=True, text=True,
        cwd=str(Path(__file__).resolve().parents[1]))
    assert proc.returncode != 0
    assert "Traceback" not in proc.stderr


def _main():
    """Run every test in this file without pytest."""
    import inspect
    import tempfile
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                if "tmp_path" in inspect.signature(fn).parameters:
                    fn(Path(tmp))
                else:
                    fn()
                print(f"   pass  {name}")
            except Exception as exc:                      # noqa: BLE001
                failed.append((name, exc))
                print(f"   FAIL  {name}: {exc}")
    print(f"\n{len(tests) - len(failed)} of {len(tests)} CLI checks passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
