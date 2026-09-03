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

# The CME reference bars in the first figure. These are the published figures of
# the companion paper, not a measurement made here, and they are hardcoded for
# exactly that reason: nothing in this package produces them and no shipped
# panel should appear to. Different asset class, period and feed.
CME_COMPANION = {"capture_bp": 0.654, "adverse_bp": -0.661, "net_bp": -0.022}


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
    """The three crypto venues, with the CME companion as a reference panel.

    The CME bars are not a fourth measurement from this study. They are the
    published companion figures on a different asset class, period and feed, so
    they are shaded and separated to keep the contrast readable without
    presenting it as a matched comparison.
    """
    d = load("decomposition_by_venue.json")["venues"]
    names = list(d)
    cap = [d[v]["pooled"]["10s_row_mean"]["capture_bp"] for v in names]
    adv = [d[v]["pooled"]["10s_row_mean"]["adverse_bp"] for v in names]
    net = [d[v]["pooled"]["10s_row_mean"]["net_bp"] for v in names]

    labels = ["CME futures"] + [DISPLAY.get(n, n) for n in names]
    cap = [CME_COMPANION["capture_bp"]] + cap
    adv = [CME_COMPANION["adverse_bp"]] + adv
    net = [CME_COMPANION["net_bp"]] + net

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.8, 3.6))
    ax.axvspan(-0.5, 0.5, color="#8a94a6", alpha=0.13, zorder=0)
    ax.axvline(0.5, color="#8a94a6", lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.bar(x - 0.22, cap, 0.2, color=C_CAP, label="captured half-spread", zorder=2)
    ax.bar(x, adv, 0.2, color=C_ADV, label="adverse selection", zorder=2)
    ax.bar(x + 0.22, net, 0.2, color=C_NET, label="net markout", zorder=2)
    ax.axhline(0, color="#2d3748", lw=0.9, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_ylabel("basis points per fill")
    ax.set_title("Adverse selection cancels the touch on CME futures and\n"
                 "over-consumes it on crypto perpetuals", loc="left")
    ax.text(0, 0.035, "reference panel,\ncompanion paper", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=6.5, color="#4a5568")
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              frameon=False, handlelength=1.4, columnspacing=1.6)
    save(fig, "fig1_decomposition")



RULE_STYLE = {"lee_ready": ("Lee-Ready", "#2F6B8C"), "tick": ("tick rule", "#A5722C")}


def fig_sign_rule():
    """What a trade-sign classifier costs, and why no constant fixes it.

    Left: the net error each rule makes against the exchange flag, with its
    paired interval. The two venues sit on opposite sides of zero, which is
    the whole point: there is no correction to hand anyone. Right: the same
    rules' per-coin agreement, which runs with the tick grid and is what makes
    the venues differ.
    """
    d = load("sign_rule_counterfactual.json")["venues"]
    venues = list(d)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.4, 3.5),
                                 gridspec_kw={"width_ratios": [1.15, 1]})

    x = np.arange(len(venues))
    for i, (rule, (label, colour)) in enumerate(RULE_STYLE.items()):
        off = (i - 0.5) * 0.34
        vals = [d[v]["by_rule"][rule]["net_error_bp_10s"] for v in venues]
        ci = [d[v]["net_error_10s"][rule] for v in venues]
        lo = [c["ci95_t_two_way"][0] if c["ci95_t_two_way"] else c["ci95_t"][0] for c in ci]
        hi = [c["ci95_t_two_way"][1] if c["ci95_t_two_way"] else c["ci95_t"][1] for c in ci]
        ax.bar(x + off, vals, 0.3, color=colour, label=label, zorder=2)
        ax.errorbar(x + off, vals, yerr=[np.array(vals) - np.array(lo),
                                         np.array(hi) - np.array(vals)],
                    fmt="none", ecolor="#2d3748", elinewidth=1.0, capsize=3, zorder=3)
        for j, (xi, val) in enumerate(zip(x + off, vals)):
            share = 100 * val / abs(d[venues[j]]["by_rule"]["true"]["net_bp_10s"])
            tip = hi[j] if val > 0 else lo[j]
            ax.annotate(f"{share:+.1f}%", (xi, tip), textcoords="offset points",
                        xytext=(0, 5 if val > 0 else -12), ha="center",
                        fontsize=7, color="#4a5568")
    ax.axhline(0, color="#2d3748", lw=0.9, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY.get(v, v) for v in venues])
    ax.set_ylabel("net error against the exchange flag, bp")
    ax.set_title("The error changes sign between venues", loc="left", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    lim = max(abs(np.array(ax.get_ylim()))) * 1.25
    ax.set_ylim(-lim, lim)

    for i, (rule, (label, colour)) in enumerate(RULE_STYLE.items()):
        for j, v in enumerate(venues):
            by_coin = d[v]["by_rule"][rule].get("sign_accuracy_by_coin", {})
            vals = np.array(sorted(by_coin.values())) * 100
            if not vals.size:
                continue
            y = j + (i - 0.5) * 0.34
            bx.plot([vals.min(), vals.max()], [y, y], color=colour, lw=1.0,
                    alpha=0.55, zorder=2, solid_capstyle="round")
            bx.scatter(vals, np.full(vals.size, y), s=9, color=colour,
                       alpha=0.7, linewidths=0, zorder=3,
                       label=label if j == 0 else None)
            pooled = d[v]["by_rule"][rule]["sign_accuracy"] * 100
            bx.scatter([pooled], [y], marker="|", s=160, color="#2d3748",
                       linewidths=1.4, zorder=4)
    bx.set_yticks(np.arange(len(venues)))
    bx.set_yticklabels([DISPLAY.get(v, v) for v in venues])
    bx.set_ylim(len(venues) - 0.55, -0.45)
    bx.set_xlabel("trades signed as the exchange did, percent\n"
                  "one dot per coin, the dark rule marks the pooled figure", fontsize=8.5)
    bx.set_title("Agreement runs with the tick grid", loc="left", fontsize=9.5)
    bx.legend(fontsize=7.5, frameon=False, loc="upper left", handletextpad=0.3)
    bx.margins(x=0.06)
    fig.subplots_adjust(wspace=0.28)
    save(fig, "fig5_sign_rule")

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
                 label=("touch (L1)" if level == 1 else
                        f"deepest (L{level})" if level == len(lv) else
                        "levels 2 to 4" if level == 2 else None))
    r_min = d["min_rebate_for_any_level_profitable_bp"]
    ax2.axvspan(min(rebs), r_min, color="#718096", alpha=0.10, lw=0)
    ax2.axvline(d["touch_overtakes_deepest_at_rebate_bp"], color=C_ACC,
                lw=1.4, ls="--")
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




# The interval-calibration figure. It visualises the coverage sweep behind the
# calibration section, whose row order the shipped sweep does not itself carry:
# figure and table should read the same way round.
CALIB_ORDER = [
    ("cluster count alone", [
        ("8 clusters", "8 clusters"),
        ("11 clusters", "11 clusters"),
        ("13 clusters", "13 clusters"),
        ("20 clusters", "20 clusters"),
        ("8 clusters, 300 rows each", "8 clusters, 300 rows each"),
    ]),
    ("unequal cluster weights", [
        ("13 clusters, weight skew 1.0", "weight skew 1.0"),
        ("13 clusters, weight skew 1.5", "weight skew 1.5"),
        ("13 clusters, weight skew 2.5", "weight skew 2.5"),
        ("13 clusters, weight skew 3.5", "weight skew 3.5"),
        ("13 clusters, weight at cluster level, skew 1.5", "weight at cluster level, skew 1.5"),
        ("13 clusters, heavy clusters are the atypical ones", "heavy clusters are atypical"),
    ]),
    ("cluster effects", [
        ("13 clusters, coin effect 1.0", "cluster effect 1.0"),
        ("13 clusters, coin effect 2.0", "cluster effect 2.0"),
        ("20 clusters, coin effect 1.0 and skew 1.5", "20 clusters, effect 1.0, skew 1.5"),
        ("13 clusters, right-skewed coin effects 0.8", "right-skewed effects 0.8"),
        ("13 clusters, right-skewed coin effects 1.2", "right-skewed effects 1.2"),
        ("13 clusters, right-skewed coin effects 2.0", "right-skewed effects 2.0"),
    ]),
    ("a shock common to every unit on a date", [
        ("13 clusters, daily shock 1.0", "13 clusters, common daily shock"),
        ("20 clusters, daily shock 1.0", "20 clusters, common daily shock"),
    ]),
]
CALIB_METHODS = [("percentile", "percentile", "#8a94a6"), ("t", "$t$", "#2F6B8C"),
                ("two_way", "two-way", "#3F8659"), ("wild", "sign-flip", "#A5722C")]


def fig_calibration_coverage():
    """Four interval methods against a nominal 0.05, and where they abstain.

    The point of the left panel is not any single row but the pattern: the
    two-way estimator is alone under a common daily shock and indistinguishable
    from the other three under skewed cluster effects, where all four land near
    0.42. The right panel is the abstention rule, which fires on weight
    concentration and is blind to the condition that breaks everything.
    """
    sweep = load("coverage_sweep.json")
    by_condition = {r["condition"]: r for r in sweep["rows"]}
    nominal = sweep["nominal"]

    # Each family gets its own blank row, so the family name lives in the tick
    # column instead of floating over the data.
    rows, labels, positions, header_rows, bands = [], [], [], [], []
    for family, members in CALIB_ORDER:
        header_rows.append(len(labels))
        labels.append(family)
        first = len(labels)
        for key, label in members:
            rows.append(by_condition[key])
            positions.append(len(labels))
            labels.append("   " + label)
        bands.append((first - 1, len(labels)))

    y = np.array(positions, dtype=float)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.8, 6.0), sharey=True,
                                 gridspec_kw={"width_ratios": [2.3, 1]})

    for i, (lo, hi) in enumerate(bands):
        if i % 2:
            for a in (ax, bx):
                a.axhspan(lo - 0.5, hi - 0.5, color="#8a94a6", alpha=0.09, zorder=0)

    ax.axvline(nominal, color="#A4453D", lw=1.1, ls=(0, (4, 3)), zorder=2)
    ax.annotate(f"nominal {nominal}", (nominal, -0.62), xytext=(4, 0),
                textcoords="offset points", fontsize=7.5, color="#A4453D",
                ha="left", va="center")

    for j, r in enumerate(rows):
        got = [r[k] for k, _, _ in CALIB_METHODS if r[k] is not None]
        ax.plot([min(got), max(got)], [y[j], y[j]], color="#cbd2dc", lw=0.8, zorder=2)
    for i, (key, label, colour) in enumerate(CALIB_METHODS):
        vals = [r[key] if r[key] is not None else np.nan for r in rows]
        ax.scatter(vals, y + (i - 1.5) * 0.19, s=26, color=colour, label=label,
                   zorder=3, linewidths=0)

    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    for tick in np.array(ax.get_yticklabels())[header_rows]:
        tick.set_fontstyle("italic")
        tick.set_color("#4a5568")
        tick.set_fontsize(8.5)
    ax.set_ylim(len(labels) - 0.5, -1.0)
    ax.set_xlim(0, 0.9)
    ax.set_xlabel("rejection rate of a true null")
    ax.set_title("Size of four interval methods", loc="left", fontsize=10)
    ax.legend(fontsize=8, frameon=False, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.09))

    avail = np.array([r["verdict_available_frac"] for r in rows])
    bx.barh(y, avail, 0.56, color="#2F6B8C", zorder=3)
    bx.barh(y, 1 - avail, 0.56, left=avail, color="#dfe4ea", zorder=3)
    for j, v in enumerate(avail):
        if v < 0.999:
            bx.annotate(f"{1 - v:.0%} withheld", (1.04, y[j]), ha="left",
                        va="center", fontsize=7, color="#2d3748", zorder=4)
    bx.set_xlim(0, 1.42)
    bx.set_xticks([0, 0.5, 1.0])
    bx.set_xlabel("replications returning a verdict")
    bx.set_title("Where the abstention rule fires", loc="left", fontsize=10)
    bx.spines["right"].set_visible(False)

    fig.subplots_adjust(wspace=0.10)
    save(fig, "fig6_calibration_coverage")

def main():
    print("writing figures to reproduce/figures/")
    fig_decomposition()
    fig_sign_rule()
    fig_per_coin()
    fig_depth_rebate()
    fig_conditional_net()
    fig_calibration_coverage()


if __name__ == "__main__":
    main()
