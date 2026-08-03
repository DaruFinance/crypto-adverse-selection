"""
Entry-markout decomposition for a passive quote on a perpetual order book.

A passive fill earns the half-spread between the fill price and the prevailing
mid, and then loses whatever the mid moves against the resting side over the
markout horizon. This module splits the entry markout into those two terms
and returns them per fill and aggregated over a set of fills, all in basis
points of the mid at fill time. Nothing is unwound and no exit cost is charged,
so these are markouts rather than realised profit and loss.

`adverse_over_capture` is a magnitude: how many times the captured half-spread
the adverse drift amounts to. It cannot on its own tell over-consumption from
the opposite case, where the mid moves the maker's way and the drift is a gain,
so `adverse_runs_against_maker` carries that sign separately. The distinction
is not hypothetical at fine grain: adverse drift is a gain on about 14 percent
of Hyperliquid coin-days at the 60 s horizon, though it is a loss in every
pooled cell the paper reports.

Every function takes `side` as the *maker's* side: +1 when the maker ends up
long, meaning the aggressor sold. That is the negation of the aggressor side
carried on the tape, so a tape column must be flipped before it is passed in.
Passing the aggressor side unflipped inverts the sign of every term returned.
The side is derived from the true aggressor flag on the tape, so no tick rule
or quote rule is applied anywhere in this module.
"""

from __future__ import annotations

import numpy as np

BP = 1e4


def spread_capture_bp(fill_price, mid_at_fill, side):
    side = np.asarray(side, dtype=np.float64)
    fill_price = np.asarray(fill_price, dtype=np.float64)
    mid_at_fill = np.asarray(mid_at_fill, dtype=np.float64)
    return side * (mid_at_fill - fill_price) / mid_at_fill * BP


def adverse_selection_bp(mid_at_fill, mid_at_horizon, side):
    """Signed so that side is +1 when the maker ends up long."""
    side = np.asarray(side, dtype=np.float64)
    mid_at_fill = np.asarray(mid_at_fill, dtype=np.float64)
    mid_at_horizon = np.asarray(mid_at_horizon, dtype=np.float64)
    return side * (mid_at_horizon - mid_at_fill) / mid_at_fill * BP


def base_symbol(coin, quote="USDT"):
    """Strip a venue's quote-currency suffix so coin keys compare across panels."""
    return coin[:-len(quote)] if coin.endswith(quote) and len(coin) > len(quote) else coin


def decompose(fill_price, mid_at_fill, mid_at_horizon, side, size=None):
    """Split entry markout into captured half-spread and adverse drift.

    `side` is the maker's side, +1 when the maker ends up long, and must be
    exactly +1 or -1 on every fill. `size` weights the aggregation and defaults
    to equal weight. Fills whose terms are not finite are dropped from both the
    aggregates and the fill count.
    """
    side_arr = np.asarray(side, dtype=np.float64)
    if not np.all(np.isin(side_arr, (-1.0, 1.0))):
        raise ValueError("side must be +1 (maker long) or -1 (maker short) "
                         "on every fill, with no missing values")
    capture = spread_capture_bp(fill_price, mid_at_fill, side)
    adverse = adverse_selection_bp(mid_at_fill, mid_at_horizon, side)
    net = capture + adverse
    finite = np.isfinite(capture) & np.isfinite(adverse)
    if size is None:
        weight = np.ones_like(capture)
    else:
        weight = np.asarray(size, dtype=np.float64)
        if weight.shape != capture.shape:
            raise ValueError("size must be an array matching the fill count")
        if not np.all(np.isfinite(weight)) or np.any(weight < 0):
            raise ValueError("size must be finite and non-negative")
    weight = np.where(finite, weight, 0.0)
    total = weight.sum()
    if total <= 0:
        return {
            "n_fills": 0,
            "capture_bp": float("nan"),
            "adverse_bp": float("nan"),
            "net_bp": float("nan"),
            "adverse_over_capture": float("nan"),
            "adverse_runs_against_maker": False,
            "per_fill": {"capture_bp": capture, "adverse_bp": adverse,
                         "net_bp": net},
        }
    cap = float((np.where(finite, capture, 0.0) * weight).sum() / total)
    adv = float((np.where(finite, adverse, 0.0) * weight).sum() / total)
    return {
        "n_fills": int(finite.sum()),
        "capture_bp": cap,
        "adverse_bp": adv,
        "net_bp": float((np.where(finite, net, 0.0) * weight).sum() / total),
        "adverse_over_capture": (abs(adv) / abs(cap) if cap else float("nan")),
        "adverse_runs_against_maker": bool(adv < 0),
        "per_fill": {"capture_bp": capture, "adverse_bp": adverse,
                     "net_bp": net},
    }


def net_after_fee(net_bp, maker_fee_bp):
    return float(net_bp) - float(maker_fee_bp)


def breakeven_rebate_bp(net_bp):
    return -float(net_bp)
