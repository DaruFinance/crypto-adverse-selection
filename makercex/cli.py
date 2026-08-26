"""
``makercex measure``: point the measurement at your own dump.

The rest of this package documents and tests the rule; this runs it. Give it
book snapshots and a trade tape carrying a true aggressor flag and it returns
the captured half-spread, the adverse drift, the net entry markout and a
cluster-robust interval, with the abstention rule of :mod:`makercex.inference`
firing when the effective cluster count will not support a verdict.

    makercex measure --snapshots book.parquet --trades tape.parquet --horizon 10s

The data boundary stated in the README is about *our* raw archives, which are
not redistributable here. It says nothing about any other archive, so the runner ships even
though the archives do not.

What it needs
-------------
Snapshots: a timestamp, the best bid and ask price and the size resting at each,
and optionally a symbol. Trades: a timestamp, price, size, an aggressor side and
optionally a symbol. Column names are detected from the usual spellings and can
be overridden with the ``--*-col`` flags. Timestamps may be integers in seconds,
milliseconds, microseconds or nanoseconds, or a datetime column; the unit is
inferred from magnitude unless ``--time-unit`` names it.

Aggressor side is read as +1 for an aggressive buy and -1 for an aggressive
sell, and ``buy``/``sell``, ``b``/``s`` and a ``is_buyer_maker`` boolean are all
accepted. That flag is the one ingredient this measurement cannot infer: a tick
rule or Lee-Ready in its place carries the error the paper's sign-rule section
measures, so the runner requires a true flag rather than silently classifying.

What it does
------------
Each (symbol, date) cell is simulated separately with the reference
last-in-queue touch quoter, decomposed at the requested horizon and reduced to
one fill-weighted row. Those rows are the panel the interval is taken on,
clustered on calendar month and, where a symbol column is present, on symbol as
well, which is the two-way estimator the paper's headline uses.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .decomp import breakeven_rebate_bp, decompose
from .fills import markout_mids, simulate_touch_fills
from .inference import MIN_CLUSTERS, cluster_bootstrap

_ALIASES = {
    "ts": ("ts_ns", "ts", "timestamp", "time", "datetime", "transact_time",
           "ts_event", "recv_ns", "t"),
    "bid_px": ("bid_px", "bid_price", "bid", "best_bid", "best_bid_price", "bp"),
    "bid_sz": ("bid_sz", "bid_size", "bid_qty", "bid_quantity", "best_bid_size", "bq"),
    "ask_px": ("ask_px", "ask_price", "ask", "best_ask", "best_ask_price", "ap"),
    "ask_sz": ("ask_sz", "ask_size", "ask_qty", "ask_quantity", "best_ask_size", "aq"),
    "price": ("price", "px", "trade_price", "p"),
    "size": ("size", "sz", "qty", "quantity", "amount", "q"),
    "side": ("side", "aggressor", "aggressor_side", "is_buyer_maker",
             "maker_side", "direction"),
    "symbol": ("symbol", "coin", "ticker", "instrument", "pair", "market"),
}

_UNIT_NS = {"s": 1_000_000_000, "ms": 1_000_000, "us": 1_000, "ns": 1}


class Bail(Exception):
    """A user-facing failure: printed as a message, never as a traceback."""


def parse_horizon(text):
    """'10s', '60s', '500ms', '2m' -> nanoseconds."""
    t = str(text).strip().lower()
    for suffix, mult in (("ms", 1_000_000), ("us", 1_000), ("ns", 1),
                         ("s", 1_000_000_000), ("m", 60_000_000_000),
                         ("h", 3_600_000_000_000)):
        if t.endswith(suffix):
            head = t[: -len(suffix)]
            try:
                return int(float(head) * mult)
            except ValueError:
                break
    raise Bail(f"cannot read a horizon from {text!r}; try 10s, 60s or 500ms")


def _read_table(path, columns=None):
    """Read a parquet or delimited file into {column: list}.

    Every failure here leaves as a Bail. A wrong path is the likeliest first
    mistake anyone makes with this command, and a traceback is the wrong way to
    tell them.
    """
    p = str(path)
    if not Path(p).exists():
        raise Bail(f"no such file: {p}")
    if p.endswith((".parquet", ".pq")):
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise Bail("reading parquet needs pyarrow: pip install 'makercex[data]', "
                       "or convert the file to CSV first")
        try:
            table = pq.read_table(p)
        except Exception as exc:                              # noqa: BLE001
            raise Bail(f"{p} does not read as parquet: {exc}")
        return {name: table[name].to_pylist() for name in table.schema.names}
    try:
        return _read_delimited(p)
    except OSError as exc:
        raise Bail(f"cannot read {p}: {exc}")


def _read_delimited(p):
    import csv as _csv
    with open(p, newline="") as fh:
        sample = fh.read(8192)
        fh.seek(0)
        try:
            dialect = _csv.Sniffer().sniff(sample, delimiters=",\t;|")
        except _csv.Error:
            dialect = _csv.excel
        rows = list(_csv.DictReader(fh, dialect=dialect))
    if not rows:
        raise Bail(f"{p} has a header and no rows")
    out = {k: [] for k in rows[0]}
    for r in rows:
        for k, v in r.items():
            out[k].append(v)
    return out


def _pick(table, want, override, what, required=True):
    """Resolve one logical column against the file's actual header."""
    lower = {str(k).strip().lower(): k for k in table}
    if override:
        if override in table:
            return override
        if override.lower() in lower:
            return lower[override.lower()]
        raise Bail(f"{what}: no column named {override!r}; the file has "
                   f"{', '.join(map(str, table))}")
    for cand in _ALIASES.get(want, (want,)):
        if cand in lower:
            return lower[cand]
    if not required:
        return None
    flag = {("snapshots", "ts"): "--snapshot-time-col",
            ("trades", "ts"): "--trade-time-col"}.get(
                (what, want), f"--{want.replace('_', '-')}-col")
    raise Bail(f"{what}: could not find a {want} column among "
               f"{', '.join(map(str, table))}; name it with {flag}")


def _to_ns(values, unit=None, what="timestamp"):
    """Timestamps to int64 nanoseconds, inferring the unit from magnitude."""
    first = next((v for v in values if v not in (None, "")), None)
    if first is None:
        raise Bail(f"{what} column is empty")
    if isinstance(first, datetime):
        return np.array([int(v.replace(tzinfo=v.tzinfo or timezone.utc).timestamp()
                             * 1_000_000_000) for v in values], dtype=np.int64)
    if isinstance(first, str) and not first.strip().lstrip("-").isdigit():
        out = []
        for v in values:
            try:
                d = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
            except ValueError:
                raise Bail(f"cannot read {what} {v!r} as a number or a datetime")
            out.append(int(d.replace(tzinfo=d.tzinfo or timezone.utc).timestamp()
                           * 1_000_000_000))
        return np.array(out, dtype=np.int64)
    raw = np.array([int(float(v)) for v in values], dtype=np.int64)
    if unit:
        if unit not in _UNIT_NS:
            raise Bail(f"--time-unit must be one of {', '.join(_UNIT_NS)}")
        return raw * _UNIT_NS[unit]
    # Seconds since the epoch land near 1.7e9 today and each thousandfold step
    # up is the next finer unit, so the cut for each sits two orders above the
    # value it accepts and two below the next. Inferring beats defaulting,
    # because a wrong unit silently rescales every horizon in the run.
    m = float(np.nanmedian(np.abs(raw.astype(np.float64))))
    for unit_name, below in (("s", 1e11), ("ms", 1e14), ("us", 1e17)):
        if m < below:
            return _checked(raw * _UNIT_NS[unit_name], unit_name, what)
    return _checked(raw, "ns", what)


# A wrong unit does not fail, it rescales every horizon in the run and returns a
# plausible number, so an inferred unit is checked against the calendar it
# implies. A thousandfold error moves the implied year by centuries, which is
# what makes this catchable at all. An explicit --time-unit skips the check.
_EPOCH_FLOOR_NS = 315_532_800_000_000_000    # 1980-01-01
_EPOCH_CEIL_NS = 4_102_444_800_000_000_000   # 2100-01-01


def _checked(ns, unit_name, what):
    m = float(np.nanmedian(ns.astype(np.float64))) if ns.size else 0.0
    if _EPOCH_FLOOR_NS <= m <= _EPOCH_CEIL_NS:
        return ns
    year = datetime.fromtimestamp(max(m, 0) / 1e9, tz=timezone.utc).year
    raise Bail(
        f"{what}: read as {unit_name} the tape sits in the year {year}, which "
        f"is almost certainly the wrong unit rather than the right date. Name "
        f"the unit with --time-unit to override this check.")


_TRUTHY = ("1", "+1", "t", "true", "y", "yes")
_FALSY = ("0", "-1", "f", "false", "n", "no")
_BUY = ("1", "+1", "b", "buy", "buyer", "bid", "taker_buy", "true")
_SELL = ("-1", "s", "sell", "seller", "ask", "taker_sell", "false")


def _to_side(values, column_name):
    """Aggressor side to int8 +1 (aggressive buy) / -1 (aggressive sell).

    A column named for the *maker* rather than the aggressor is inverted
    before anything else, because it answers the opposite question: where
    is_buyer_maker is true the buyer was resting, so the aggressor sold. That
    branch has to come first and cover the integer encoding as well as the
    boolean one, since the venues that publish this field ship it as 0/1 in
    their CSV archives and as a JSON boolean on their sockets. Reading a 1
    there as an aggressive buy would flip the sign on every trade in the file
    and turn captured spread into adverse selection without failing.
    """
    maker_flag = "buyer_maker" in str(column_name).lower()
    out = np.empty(len(values), dtype=np.int8)
    for i, v in enumerate(values):
        t = ("true" if v else "false") if isinstance(v, bool) else str(v).strip().lower()
        if maker_flag:
            if t in _TRUTHY:
                out[i] = -1        # buyer rested, so the aggressor sold
            elif t in _FALSY:
                out[i] = 1
            else:
                raise Bail(f"cannot read maker flag {v!r} in column "
                           f"{column_name!r}; expected a boolean or 0/1")
            continue
        if t in _BUY:
            out[i] = 1
        elif t in _SELL:
            out[i] = -1
        else:
            raise Bail(f"cannot read aggressor side {v!r} in column "
                       f"{column_name!r}; expected +1/-1, buy/sell, or a "
                       f"is_buyer_maker boolean")
    return out


def _floats(values, what):
    try:
        return np.array([float(v) for v in values], dtype=np.float64)
    except (TypeError, ValueError):
        raise Bail(f"{what} column carries a value that is not a number")


def _date_of(ts_ns):
    return datetime.fromtimestamp(int(ts_ns) / 1e9, tz=timezone.utc).strftime("%Y-%m-%d")


def load_inputs(args):
    """Both files into aligned numpy arrays, grouped by (symbol, date)."""
    snaps = _read_table(args.snapshots)
    trades = _read_table(args.trades)

    s_ts_col = _pick(snaps, "ts", args.snapshot_time_col, "snapshots")
    t_ts_col = _pick(trades, "ts", args.trade_time_col, "trades")
    if s_ts_col is None:
        raise Bail("snapshots: no timestamp column found")

    cols = {
        "s_ts": _to_ns(snaps[s_ts_col], args.time_unit, "snapshot timestamp"),
        "bid_px": _floats(snaps[_pick(snaps, "bid_px", args.bid_px_col, "snapshots")], "bid price"),
        "bid_sz": _floats(snaps[_pick(snaps, "bid_sz", args.bid_sz_col, "snapshots")], "bid size"),
        "ask_px": _floats(snaps[_pick(snaps, "ask_px", args.ask_px_col, "snapshots")], "ask price"),
        "ask_sz": _floats(snaps[_pick(snaps, "ask_sz", args.ask_sz_col, "snapshots")], "ask size"),
        "t_ts": _to_ns(trades[t_ts_col], args.time_unit, "trade timestamp"),
        "t_px": _floats(trades[_pick(trades, "price", args.price_col, "trades")], "trade price"),
        "t_sz": _floats(trades[_pick(trades, "size", args.size_col, "trades")], "trade size"),
    }
    side_col = _pick(trades, "side", args.side_col, "trades")
    cols["t_side"] = _to_side(trades[side_col], side_col)

    s_sym_col = _pick(snaps, "symbol", args.symbol_col, "snapshots", required=False)
    t_sym_col = _pick(trades, "symbol", args.symbol_col, "trades", required=False)
    n_s, n_t = cols["s_ts"].size, cols["t_ts"].size
    cols["s_sym"] = (np.array([str(v) for v in snaps[s_sym_col]]) if s_sym_col
                     else np.full(n_s, "ALL"))
    cols["t_sym"] = (np.array([str(v) for v in trades[t_sym_col]]) if t_sym_col
                     else np.full(n_t, "ALL"))
    cols["has_symbol"] = bool(s_sym_col and t_sym_col)
    return cols


def measure(cols, horizon_ns, size=1.0, n_boot=4000, seed=7):
    """Simulate, decompose and reduce to one fill-weighted row per cell."""
    s_order = np.argsort(cols["s_ts"], kind="stable")
    t_order = np.argsort(cols["t_ts"], kind="stable")
    snap = {k: cols["s_" + k if k in ("ts", "sym") else k][s_order]
            for k in ("ts", "sym", "bid_px", "bid_sz", "ask_px", "ask_sz")}
    tape = {"ts": cols["t_ts"][t_order], "sym": cols["t_sym"][t_order],
            "px": cols["t_px"][t_order], "sz": cols["t_sz"][t_order],
            "side": cols["t_side"][t_order]}

    cells = []
    symbols = sorted(set(snap["sym"]) & set(tape["sym"])) or ["ALL"]
    for sym in symbols:
        sm = snap["sym"] == sym
        tm = tape["sym"] == sym
        if not sm.any() or not tm.any():
            continue
        s_ts = snap["ts"][sm]
        s_mid = (snap["bid_px"][sm] + snap["ask_px"][sm]) * 0.5
        days = np.array([_date_of(t) for t in s_ts])
        t_days = np.array([_date_of(t) for t in tape["ts"][tm]])
        for day in sorted(set(days)):
            d = days == day
            td = t_days == day
            if d.sum() < 2 or not td.any():
                continue
            f_ts, f_side, f_px, f_sz = simulate_touch_fills(
                s_ts[d], snap["bid_px"][sm][d], snap["bid_sz"][sm][d],
                snap["ask_px"][sm][d], snap["ask_sz"][sm][d],
                tape["ts"][tm][td], tape["px"][tm][td], tape["sz"][tm][td],
                tape["side"][tm][td], size=size)
            if f_ts.size == 0:
                continue
            m0, mt = markout_mids(f_ts, s_ts[d], s_mid[d], horizon_ns)
            got = decompose(f_px, m0, mt, f_side)
            if got["n_fills"] == 0:
                continue
            cells.append({"symbol": sym, "date": day, "n_fills": got["n_fills"],
                          "capture_bp": got["capture_bp"],
                          "adverse_bp": got["adverse_bp"],
                          "net_bp": got["net_bp"]})
    if not cells:
        raise Bail("no fills: the quoter sits behind all displayed size and "
                   "fills only on the spill above it, so check that the trade "
                   "prices match the quoted prices and that the aggressor side "
                   "is the aggressor's rather than the maker's")

    w = np.array([c["n_fills"] for c in cells], dtype=np.float64)
    pooled = {k: float((np.array([c[k] for c in cells]) * w).sum() / w.sum())
              for k in ("capture_bp", "adverse_bp", "net_bp")}
    months = [c["date"][:7] for c in cells]
    coins = [c["symbol"] for c in cells]
    net = np.array([c["net_bp"] for c in cells], dtype=np.float64)
    two_way = cols.get("has_symbol") and len(set(coins)) > 1
    ci = cluster_bootstrap(net, w, months, n_boot=n_boot, seed=seed,
                           cluster_b=coins if two_way else None)
    return {
        "n_cells": len(cells), "n_fills": int(w.sum()),
        "n_symbols": len(set(coins)), "n_months": len(set(months)),
        "pooled": pooled,
        "breakeven_rebate_bp": breakeven_rebate_bp(pooled["net_bp"]),
        "interval": ci, "clustering": "month and symbol" if two_way else "month",
        "cells": cells,
    }


def _report(res, horizon, out=sys.stdout):
    ci = res["interval"]
    p = res["pooled"]
    w = out.write
    w(f"\nEntry markout at the touch, {horizon} horizon, pre-fee\n")
    w(f"  {res['n_fills']:,} fills over {res['n_cells']:,} symbol-days "
      f"({res['n_symbols']} symbols, {res['n_months']} months)\n\n")
    w(f"  captured half-spread   {p['capture_bp']:+9.4f} bp\n")
    w(f"  adverse selection      {p['adverse_bp']:+9.4f} bp\n")
    w(f"  net entry markout      {p['net_bp']:+9.4f} bp\n")
    w(f"  break-even rebate      {res['breakeven_rebate_bp']:9.4f} bp\n\n")
    w(f"  clustered on {res['clustering']}, {ci['n_clusters']} clusters, "
      f"{ci['n_effective_clusters']:.1f} effective\n")
    if ci["verdict_is_available"]:
        lo, hi = ci["ci95_t_two_way"] or ci["ci95_t"]
        w(f"  95% interval           [{lo:+.4f}, {hi:+.4f}]\n")
        w(f"  clears zero            {'yes' if ci['clears_zero'] else 'no'}\n")
    else:
        w(f"  95% interval           withheld\n")
        w(f"  The effective cluster count is below {MIN_CLUSTERS}, which is the\n"
          f"  floor at which an interval here is worth reading. That is a\n"
          f"  withheld verdict, not a failed test.\n")
    w("\n  This is a one-sided entry markout, not realised profit: nothing is\n"
      "  unwound and no exit cost is charged.\n\n")


def build_parser():
    ap = argparse.ArgumentParser(
        prog="makercex",
        description="Entry-markout decomposition on your own book and tape.")
    sub = ap.add_subparsers(dest="command", required=True)
    m = sub.add_parser("measure", help="measure entry markout on a book and tape")
    m.add_argument("--snapshots", required=True,
                   help="book snapshots (parquet or delimited text)")
    m.add_argument("--trades", required=True,
                   help="trade tape with a true aggressor flag")
    m.add_argument("--horizon", default="10s", help="markout horizon (default 10s)")
    m.add_argument("--size", type=float, default=1.0, help="quoted size (default 1.0)")
    m.add_argument("--n-boot", type=int, default=4000, help="bootstrap resamples")
    m.add_argument("--seed", type=int, default=7)
    m.add_argument("--json", help="also write the full result here")
    m.add_argument("--time-unit", choices=sorted(_UNIT_NS),
                   help="timestamp unit; inferred from magnitude if omitted")
    for flag, helptext in (
            ("snapshot-time-col", "timestamp column in the snapshot file"),
            ("trade-time-col", "timestamp column in the trade file"),
            ("bid-px-col", "best bid price column"), ("bid-sz-col", "best bid size column"),
            ("ask-px-col", "best ask price column"), ("ask-sz-col", "best ask size column"),
            ("price-col", "trade price column"), ("size-col", "trade size column"),
            ("side-col", "aggressor side column"),
            ("symbol-col", "symbol column, present in both files or neither")):
        m.add_argument(f"--{flag}", help=helptext)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        res = measure(load_inputs(args), parse_horizon(args.horizon),
                      size=args.size, n_boot=args.n_boot, seed=args.seed)
    except Bail as exc:
        print(f"makercex measure: {exc}", file=sys.stderr)
        return 2
    _report(res, args.horizon)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(res, fh, indent=2, default=float)
        print(f"  full result -> {args.json}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
