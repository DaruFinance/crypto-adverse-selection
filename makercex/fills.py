"""
Reference implementation of the posting and fill rule behind every panel here.

The README describes this rule in prose under "Measurement rule". Prose cannot be
checked. This is the same rule as code, and the smoke test asserts each property
the prose claims.

The quoter posts one unit on each side at the best bid and the best ask, and
only when both sides are present. It re-posts at every snapshot. An order lives for
exactly one snapshot interval and priority never carries over. At placement the
order sits behind the entire visible size resting at its price, which puts it
last in queue.

A trade fills the order only when three things hold. The aggressor must be on
the opposite side. The trade price must match the quote price to within `eps`,
and the trade must be larger than the queue still ahead.

The fill is the spill above that queue, capped at the size still resting. Queue
ahead is drawn down by whatever each trade consumes inside the interval.

That reference mid enters the captured leg and the adverse leg with opposite
signs, so it cancels in the net and only moves the boundary between them. A
staler reference books pre-fill drift into the adverse leg and inflates the
measured capture, which makes the split sensitive to re-quote spacing in a way
the net is not.

Mid prices are read on a step rule. `mid_at_or_before` carries the mid of the
last snapshot at or before the time asked for. It does not interpolate, and it
is undefined before the first snapshot. `markout_mids` applies that rule at the
fill and again one horizon later.

The order posted at the final snapshot has no following snapshot to retire it,
so it stands until the tape ends. That follows the measurement path rather than
the one-interval description, and on a day carrying hundreds of thousands of
intervals it touches one of them.

Two guards carry over from the measurement path. A trade timestamped before the
posting instant cannot fill the order, and a remainder below `size * DUST_FRAC`
counts as consumed rather than resting, which stops an exactly filled order from
emitting a further row of no economic size.

This reference orders events by timestamp alone, where the measurement path also
carries a receive time to break ties. Ties are resolved here by treating the
snapshot as posted before any trade sharing its timestamp.
"""

from __future__ import annotations

import numpy as np

DUST_FRAC = 1e-9
PX_EPSILON = 1e-9


def mid_at_or_before(snap_ts, snap_mid, when):
    """Mid carried by the last snapshot at or before each time, NaN before the first."""
    snap_ts = np.asarray(snap_ts, dtype=np.int64)
    snap_mid = np.asarray(snap_mid, dtype=np.float64)
    if snap_ts.size != snap_mid.size:
        raise ValueError("snapshot time and mid arrays must share one length")
    when = np.asarray(when, dtype=np.int64)
    idx = np.searchsorted(snap_ts, when, side="right") - 1
    out = np.where(idx < 0, np.nan, snap_mid[np.clip(idx, 0, None)])
    return out


def markout_mids(fill_ts, snap_ts, snap_mid, horizon_ns):
    """The mid at each fill and the mid one horizon later, on the same step rule."""
    fill_ts = np.asarray(fill_ts, dtype=np.int64)
    return (mid_at_or_before(snap_ts, snap_mid, fill_ts),
            mid_at_or_before(snap_ts, snap_mid, fill_ts + int(horizon_ns)))


def simulate_touch_fills(snap_ts, bid_px, bid_sz, ask_px, ask_sz,
                         trd_ts, trd_px, trd_sz, trd_side,
                         size=1.0, eps=PX_EPSILON):
    """Return the fills a one-unit touch quoter takes, as (ts, side, px, sz)."""
    snap_ts = np.asarray(snap_ts, dtype=np.int64)
    bid_px = np.asarray(bid_px, dtype=np.float64)
    bid_sz = np.asarray(bid_sz, dtype=np.float64)
    ask_px = np.asarray(ask_px, dtype=np.float64)
    ask_sz = np.asarray(ask_sz, dtype=np.float64)
    trd_ts = np.asarray(trd_ts, dtype=np.int64)
    trd_px = np.asarray(trd_px, dtype=np.float64)
    trd_sz = np.asarray(trd_sz, dtype=np.float64)
    trd_side = np.asarray(trd_side, dtype=np.int8)
    lens = {snap_ts.size, bid_px.size, bid_sz.size, ask_px.size, ask_sz.size}
    if len(lens) != 1:
        raise ValueError("snapshot arrays must share one length")
    lens = {trd_ts.size, trd_px.size, trd_sz.size, trd_side.size}
    if len(lens) != 1:
        raise ValueError("trade arrays must share one length")
    if size <= 0:
        raise ValueError("size must be positive")
    if np.any((trd_side != 1) & (trd_side != -1)):
        raise ValueError("trade side must be +1 for a buy or -1 for a sell")
    if np.any(np.diff(snap_ts) < 0) or np.any(np.diff(trd_ts) < 0):
        raise ValueError("snapshot and trade times must be sorted ascending")

    n_snap, n_trd = snap_ts.size, trd_ts.size
    f_ts, f_side, f_px, f_sz = [], [], [], []
    dust = size * DUST_FRAC
    ti = 0
    for si in range(n_snap):
        start = snap_ts[si]
        end = snap_ts[si + 1] if si + 1 < n_snap else np.iinfo(np.int64).max
        while ti < n_trd and trd_ts[ti] < start:
            ti += 1
        both = (np.isfinite(bid_px[si]) and np.isfinite(ask_px[si])
                and bid_sz[si] > 0 and ask_sz[si] > 0)
        if not both:
            while ti < n_trd and trd_ts[ti] < end:
                ti += 1
            continue
        b_rem, a_rem = size, size
        b_queue, a_queue = bid_sz[si], ask_sz[si]
        while ti < n_trd and trd_ts[ti] < end:
            tside, tpx, tsz = trd_side[ti], trd_px[ti], trd_sz[ti]
            if tside == -1 and b_rem > dust and abs(tpx - bid_px[si]) <= eps:
                consumed = min(tsz, b_queue)
                b_queue -= consumed
                spill = tsz - consumed
                fill = min(spill, b_rem)
                if fill > dust:
                    f_ts.append(int(trd_ts[ti]))
                    f_side.append(1)
                    f_px.append(float(bid_px[si]))
                    f_sz.append(float(fill))
                    b_rem -= fill
            elif tside == 1 and a_rem > dust and abs(tpx - ask_px[si]) <= eps:
                consumed = min(tsz, a_queue)
                a_queue -= consumed
                spill = tsz - consumed
                fill = min(spill, a_rem)
                if fill > dust:
                    f_ts.append(int(trd_ts[ti]))
                    f_side.append(-1)
                    f_px.append(float(ask_px[si]))
                    f_sz.append(float(fill))
                    a_rem -= fill
            ti += 1
    return (np.array(f_ts, dtype=np.int64), np.array(f_side, dtype=np.int8),
            np.array(f_px, dtype=np.float64), np.array(f_sz, dtype=np.float64))
