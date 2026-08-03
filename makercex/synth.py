"""
Synthetic tape generator for the smoke test.

Produces a mid-price path, a passive quote at a chosen half-spread and fills
whose arrival is biased toward the side the mid is about to move against.

The `informed` parameter is the probability that an arriving aggressor trades
in the direction the mid is about to move. It is not itself the informed share
of flow: a fraction `2 * informed - 1` of arrivals is informed and the
remainder is noise, so `informed = 0.5` is pure noise and `informed = 1.0` is
fully informed. Below 0.5 the aggressor is the one being picked off, which
reverses the sign of adverse selection and is outside the regime the paper
describes. Raising `informed` above 0.5 drives adverse selection past the
captured half-spread and turns net entry markout negative, which is the mechanism
the paper measures. The closed form for a symmetric random walk is `adverse_bp
= -(2 * informed - 1) * vol_bp * sqrt(2 / pi)`.

The side returned is the maker's side, already flipped from the aggressor side,
so it can be passed straight to `decompose`. No real market data is used or
required.
"""

from __future__ import annotations

import numpy as np


def make_tape(n_events=20000, half_spread_bp=1.0, informed=0.85,
              vol_bp=2.0, seed=0):
    rng = np.random.default_rng(seed)
    steps = rng.standard_normal(n_events) * vol_bp / 1e4
    mid = 100.0 * np.exp(np.cumsum(steps))
    future = np.roll(mid, -1)
    future[-1] = mid[-1]
    drift_up = future > mid
    aggressor_buys = np.where(
        rng.random(n_events) < informed, drift_up, ~drift_up)
    side = np.where(aggressor_buys, -1.0, 1.0)
    fill_price = np.where(aggressor_buys,
                          mid * (1 + half_spread_bp / 1e4),
                          mid * (1 - half_spread_bp / 1e4))
    return {
        "fill_price": fill_price,
        "mid_at_fill": mid,
        "mid_at_horizon": future,
        "side": side,
        "size": np.ones(n_events),
    }


def make_panel(n_coins=6, n_dates=12, n_events=4000, seed=0,
               informed_low=0.88, informed_high=0.96, date_shock_sd=0.0,
               weight_skew=0.0):
    """Build a coin by date panel of synthetic tapes.

    `date_shock_sd` adds a shock shared by every coin on the same date, which is
    the dependence a one-way coin-clustered interval cannot see. `weight_skew`
    is the log-normal spread of fill counts across cells, which is what drives
    the effective cluster count below the raw one.
    """
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal(n_dates) * date_shock_sd
    rows = []
    for c in range(n_coins):
        informed = informed_low + (informed_high - informed_low) * rng.random()
        half_spread = 0.7 + 0.8 * rng.random()
        for d in range(n_dates):
            n = n_events
            if weight_skew:
                n = max(50, int(n_events * np.exp(
                    rng.standard_normal() * weight_skew)))
            tape = make_tape(n_events=n, half_spread_bp=half_spread,
                             informed=informed, vol_bp=2.0 + shocks[d],
                             seed=int(rng.integers(1 << 30)))
            rows.append({"coin": f"COIN{c:02d}",
                         "date": f"2025{(d % 12) + 1:02d}{(d // 12) + 1:02d}",
                         **tape})
    return rows
