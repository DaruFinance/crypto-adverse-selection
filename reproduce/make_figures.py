"""
Regenerate the paper figures from the aggregated result files.

Run from the repository root:

    python reproduce/make_figures.py

Figures are written to reproduce/figures/ as PDF and PNG.

The PDF creation date is suppressed so a rerun does not leave a reader with a
dirty working tree over a timestamp. Both formats then reproduce byte for byte
on a given matplotlib version, which the PDFs embed.

Binance keys its coins with a USDT suffix and the other two venues do not, so
the second figure strips it to keep the three panels reading on one scale.

The interval whiskers in the second figure are faded where that interval could
not have reached zero, which is 57 of the 96 cells across both horizons. A
solid whisker marks a cell whose interval had the power to fail.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from makercex import base_symbol

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
C_CAP, C_ADV, C_NET, C_ACC = "#2F6B8C", "#A4453D", "#3F8659", "#A5722C"
DISPLAY = {"bybit_perp": "Bybit", "binance_um": "Binance",
           "hyperliquid": "Hyperliquid", "asia": "Asia", "europe": "Europe",
           "us": "US"}


def load(name):
    return json.loads((HERE / name).read_text())


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=200, bbox_inches="tight",
                    metadata={"CreationDate": None} if ext == "pdf" else None)
    plt.close(fig)
    print(f"  {stem}")


def fig_decomposition():
    d = load("decomposition_by_venue.json")["venues"]
    names = list(d)
    cap = [d[v]["pooled"]["10s_row_mean"]["capture_bp"] for v in names]
    adv = [d[v]["pooled"]["10s_row_mean"]["adverse_bp"] for v in names]
    net = [d[v]["pooled"]["10s_row_mean"]["net_bp"] for v in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(x - 0.22, cap, 0.2, color=C_CAP, label="captured half-spread")
    ax.bar(x, adv, 0.2, color=C_ADV, label="adverse selection")
    ax.bar(x + 0.22, net, 0.2, color=C_NET, label="net markout")
    ax.axhline(0, color="#2d3748", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY.get(n, n) for n in names])
    ax.set_ylabel("basis points per fill")
    ax.set_title("Entry markout at the touch, 10 s horizon, pre-fee", loc="left")
    ax.legend(fontsize=8)
    save(fig, "fig1_decomposition")


def fig_per_coin():
    d = load("decomposition_by_venue.json")["venues"]
    power = load("per_coin_intervals.json")["venues"]
    fig, axes = plt.subplots(1, len(d), figsize=(11.5, 3.4))
    for ax, (venue, v) in zip(np.atleast_1d(axes), d.items()):
        pc = v["per_coin"]
        coins = sorted(pc, key=lambda c: pc[c]["net_bp"])
        labels = [base_symbol(c) for c in coins]
        vals = [pc[c]["net_bp"] for c in coins]
        lo = [pc[c]["net_ci95"][0] for c in coins]
        hi = [pc[c]["net_ci95"][1] for c in coins]
        y = np.arange(len(coins))
        cells = power.get(venue, {}).get("per_coin", {})
        strong = [bool(cells.get(c, {}).get("10s", {})
                       .get("percentile_interval_has_power_to_fail")) for c in coins]
        ax.barh(y, vals, color=[C_NET if x > 0 else C_ADV for x in vals])
        for i in range(len(coins)):
            ax.plot([lo[i], hi[i]], [y[i], y[i]], color="#2d3748",
                    lw=1.0 if strong[i] else 0.6,
                    alpha=1.0 if strong[i] else 0.35)
        ax.axvline(0, color="#2d3748", lw=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=6.5)
        ax.set_ylim(-0.6, len(coins) - 0.4)
        ax.set_title(DISPLAY.get(venue, venue), loc="left", fontsize=9)
        ax.set_xlabel("net markout, bp")
    axes[0].set_ylabel("faded whisker: interval could not reach zero",
                       fontsize=6.5)
    fig.subplots_adjust(wspace=0.42)
    save(fig, "fig2_per_coin")


def fig_depth_rebate():
    d = load("depth_rebate_frontier.json")
    lv = np.arange(1, len(d["markout_bp_by_level"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.4))
    ax = axes[0]
    ax.plot(lv, d["markout_bp_by_level"], "o-", color=C_ADV)
    ax.axhline(0, color="#2d3748", lw=0.9)
    ax.set_xticks(lv)
    ax.set_xlabel("price level from mid, 1 is the touch")
    ax.set_ylabel("net markout per fill, bp", color=C_ADV)
    axb = ax.twinx()
    axb.plot(lv, d["fill_rate_by_level"], "s--", color=C_CAP)
    axb.set_ylabel("fill rate", color=C_CAP, labelpad=2)
    axb.grid(False)
    ax.set_title("Quoting deeper improves each fill and costs the fills",
                 loc="left", fontsize=9)
    ax2 = axes[1]
    rebs = sorted(float(k) for k in d["by_rebate"])
    for i, level in enumerate(lv):
        ev = [d["by_rebate"][f"{r:.3f}"]["ev_by_level"][i] for r in rebs]
        ax2.plot(rebs, ev, lw=1.8 if level in (1, len(lv)) else 0.9,
                 color=C_NET if level == 1 else
                 (C_ADV if level == len(lv) else "#cbd5e0"),
                 label=("touch" if level == 1 else
                        f"deepest (L{level})" if level == len(lv) else None))
    r_min = d["min_rebate_for_any_level_profitable_bp"]
    ax2.axvspan(min(rebs), r_min, color="#718096", alpha=0.10, lw=0)
    ax2.axvline(d["touch_overtakes_deepest_at_rebate_bp"], color=C_ACC,
                lw=1.4, ls="--")
    ax2.axvline(d["bybit_best_published_tier_bp"], color="#718096", lw=0.9,
                ls=":")
    ax2.axhline(0, color="#2d3748", lw=0.9)
    ax2.set_xlabel("maker rebate, bp")
    ax2.set_ylabel("bp per quoting opportunity")
    ax2.set_title("Three regions: do not quote, quote deep, quote the touch",
                  loc="left", fontsize=9)
    ax2.legend(fontsize=7.5)
    fig.subplots_adjust(wspace=0.58)
    save(fig, "fig3_depth_rebate")


def fig_conditional_net():
    blocks = ("asia", "europe", "us")
    pos, tot, ps = [], [], []
    for b in blocks:
        c = load(f"conditional_net_{b}.json")
        pos.append(c["n_coins_positive"])
        tot.append(c["n_coins"])
        ps.append(c["recommended_p"])
    x = np.arange(len(blocks))
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.bar(x, [p / t for p, t in zip(pos, tot)], 0.5, color=C_CAP)
    ax.axhline(0.5, color="#2d3748", lw=0.9, ls="--")
    for i, (p, t, pv) in enumerate(zip(pos, tot, ps)):
        ax.annotate(f"{p}/{t}\np={pv:.2f}", (i, p / t), ha="center",
                    va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY.get(b, b) for b in blocks])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("share of coins with a positive contrast")
    ax.set_title("Net entry markout, high volatility minus low", loc="left",
                 fontsize=9)
    save(fig, "fig4_conditional_net")


def main():
    print("writing figures to reproduce/figures/")
    fig_decomposition()
    fig_per_coin()
    fig_depth_rebate()
    fig_conditional_net()


if __name__ == "__main__":
    main()
