"""
plot_frontier -- two panels: the calibration, and the risk/return frontier.

Palette is the same validated categorical pair used elsewhere: slot 1 blue
#2a78d6, slot 2 orange #eb6834, on the light surface #fcfcfb. Checked with the
data-viz validator -- adjacent CVD deltaE 24.7 (protan), normal-vision 33.6, both
slots above 3:1 contrast. Colour is never the only encoding: both charts carry a
legend AND direct labels.

One measure per axis. No dual axes.
"""

import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "reports")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8984"
S1 = "#2a78d6"      # blue   -- Avellaneda-Stoikov
S2 = "#eb6834"      # orange -- naive fixed spread
GRID = "#e4e3df"


def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=12, fontweight="600", loc="left", pad=12)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.tick_params(colors=INK_2, labelsize=8.5, length=0)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, HERE)}")


def fig_calibration(cal, rows):
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    xs = [d for d, r in rows if r > 0]
    ys = [math.log(r) for _, r in rows if r > 0]

    ax.plot(xs, ys, "o", color=S1, markersize=8, markeredgecolor=SURFACE,
            markeredgewidth=1.5, zorder=3, label="measured")
    fit_x = [min(xs), max(xs)]
    fit_y = [math.log(cal["A"]) - cal["k"] * x for x in fit_x]
    ax.plot(fit_x, fit_y, color=S2, linewidth=2.0, linestyle="--", zorder=2,
            label="fitted exponential")

    _style(ax, "Fill intensity decays exponentially, as the model assumes",
           "Quote distance from mid (ticks)", "log fill intensity (units/step)")
    ax.text(0.985, 0.93,
            f"$\\lambda(d) = {cal['A']:.3f}\\,e^{{-{cal['k']:.3f}d}}$\n"
            f"k = {cal['k']:.3f},  $R^2$ = {cal['r2']:.3f}\n"
            f"(k was guessed at 1.5)",
            transform=ax.transAxes, color=INK_2, fontsize=9,
            va="top", ha="right", linespacing=1.6)
    leg = ax.legend(loc="lower left", frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK_2)
    _save(fig, "fig1_calibration.png")


def fig_frontier(points):
    a_s = sorted([p for p in points if p["label"] == "A-S"], key=lambda p: p["inv_sd"])
    nv = sorted([p for p in points if p["label"] == "naive"], key=lambda p: p["inv_sd"])

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.axhline(0.0, color=MUTED, linewidth=1.2, zorder=1)

    # The x-axis is CLIPPED to where the two families actually overlap. Naive
    # half-spread=1 sits at (99.2, -10,625), and stretching the axis to reach it
    # compressed every point that matters into the left third. It is named in
    # the caption instead of drawn. Labels alternate above/below because at the
    # default single offset the 17-28 band was an unreadable pile.
    XMAX = 32.0

    for pts, colour, name, base_dy in ((a_s, S1, "Avellaneda-Stoikov", 11),
                                       (nv, S2, "Naive fixed spread", -17)):
        shown = [p for p in pts if p["inv_sd"] <= XMAX]
        ax.plot([p["inv_sd"] for p in shown], [p["pnl"] for p in shown],
                "-o", color=colour, linewidth=2.0, markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3, label=name)
        for i, p in enumerate(shown):
            dy = base_dy if i % 2 == 0 else base_dy + (13 if base_dy > 0 else -13)
            ax.annotate(f"{p['param']:g}", (p["inv_sd"], p["pnl"]),
                        textcoords="offset points", xytext=(0, dy),
                        ha="center", fontsize=7.5, color=colour)

    _style(ax, "The naive fixed spread dominates the closed form",
           "Inventory risk  (sd of position)", "P&L over 6,000 steps")
    ax.set_xlim(0, XMAX)
    ax.margins(y=0.18)
    ax.text(0.985, 0.05,
            "Labels are each family's own risk knob: $\\gamma$ for A-S,\n"
            "half-spread for naive. Naive wins at 7 of 7 of A-S's\n"
            "risk levels. Naive half-spread=1 is off-chart right,\n"
            "at inventory sd 99 and P&L $-$10,625.",
            transform=ax.transAxes, color=INK_2, fontsize=9,
            va="bottom", ha="right", linespacing=1.5)
    leg = ax.legend(loc="upper right", frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK_2)
    _save(fig, "fig2_frontier.png")


def fig_adverse(points):
    """Why naive wins: it quotes far enough out to stop being picked off."""
    a_s = sorted([p for p in points if p["label"] == "A-S"], key=lambda p: p["fills"])
    nv = sorted([p for p in points if p["label"] == "naive"], key=lambda p: p["fills"])

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.axhline(0.0, color=MUTED, linewidth=1.2, zorder=1)
    ax.plot([p["fills"] for p in a_s], [p["adverse"] for p in a_s], "-o",
            color=S1, linewidth=2.0, markersize=7, markeredgecolor=SURFACE,
            markeredgewidth=1.5, zorder=3, label="Avellaneda-Stoikov")
    ax.plot([p["fills"] for p in nv], [p["adverse"] for p in nv], "-o",
            color=S2, linewidth=2.0, markersize=7, markeredgecolor=SURFACE,
            markeredgewidth=1.5, zorder=3, label="Naive fixed spread")

    _style(ax, "Adverse selection is the whole story",
           "Fills over 6,000 steps", "Adverse selection (P&L units)")
    ax.text(0.02, 0.06,
            "Quoting wide enough to trade ~40 times instead of ~400\n"
            "removes adverse selection entirely. A-S cannot get there:\n"
            "its spread is pinned by the closed form.",
            transform=ax.transAxes, color=INK_2, fontsize=9,
            va="bottom", ha="left", linespacing=1.5)
    leg = ax.legend(loc="lower right", frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK_2)
    _save(fig, "fig3_adverse_selection.png")


def main():
    with open(os.path.join(OUT, "frontier.json"), encoding="utf-8") as fh:
        data = json.load(fh)
    print("figures:")
    fig_calibration(data["calibration"], data["calibration_rows"])
    fig_frontier(data["points"])
    fig_adverse(data["points"])


if __name__ == "__main__":
    main()
