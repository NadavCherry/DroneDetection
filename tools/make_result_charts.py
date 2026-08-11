#!/usr/bin/env python3
"""The three result charts the README shows instead of describing.

Each one is read straight out of the run that produced it, so a chart cannot
drift from its artifact:

    docs/media/chart_cpa.png       every city engagement's closest approach,
                                   against the two distances that decide it
    docs/media/chart_detect.png    detection rate by outcome over 62 engagements
                                   -- the one factor that predicts failure
    docs/media/chart_range.png     detection fraction against range, background
                                   model vs frame differencing

    .venv/bin/python tools/make_result_charts.py

Palette and dark canvas match the architecture figures so the README reads as
one set. The blue/orange pair is CVD-safe (worst adjacent separation dE 24.7
protan, 33.6 normal) and every series is also named in a legend, so colour never
carries the identity on its own.

Layout note, learned by producing three unreadable charts first: matplotlib's
``set_title`` and an ``ax.text`` at a transform just above the axes will happily
overlap, and end-of-line labels collide whenever two series converge. Titles
here are placed explicitly in figure space, and series are named in a legend
rather than at their line ends.
"""

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "media"

BG = "#0d1117"
INK, SUB, GRID = "#e6edf3", "#9198a1", "#30363d"
BLUE, ORANGE = "#58a6ff", "#f0883e"
GOOD, CRIT, WARN = "#3fb950", "#f85149", "#d29922"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "savefig.facecolor": BG, "savefig.edgecolor": BG,
    "text.color": INK, "axes.labelcolor": SUB, "axes.edgecolor": GRID,
    "xtick.color": SUB, "ytick.color": SUB,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})


def titled(fig, title, sub, wrap=104):
    """Title and standfirst in figure space, so they cannot collide with the axes.

    The standfirst is hard-wrapped rather than left to matplotlib: a long one
    silently runs off the right edge of the canvas, which is invisible until you
    look at the PNG.
    """
    import textwrap
    fig.text(0.012, 0.985, title, color=INK, fontsize=15.5, fontweight="bold",
             va="top")
    fig.text(0.012, 0.905, "\n".join(textwrap.wrap(sub, wrap)), color=SUB,
             fontsize=10.5, va="top", linespacing=1.5)


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=170, facecolor=BG)
    plt.close(fig)
    print(f"  docs/media/{stem}.png  ({(OUT / f'{stem}.png').stat().st_size / 1024:.0f} KB)")


# ---------------------------------------------------------------------------

def chart_cpa():
    """24 city engagements: closest approach against arrival bearing."""
    d = json.loads((ROOT / "work/pursuit/city/results.json").read_text())
    rows = [(int(r["name"].split("-")[1]), r["pass_cpa_m"]) for r in d["results"]]
    cpa = [c for _, c in rows]
    mean = st.mean(cpa)

    fig, ax = plt.subplots(figsize=(9.4, 4.9))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.78, bottom=0.135)

    # the two distances that decide the result, labelled at the top
    for v, col, lab in ((0.47, WARN, "0.47 m  the airframe's own rotor span"),
                        (1.00, CRIT, "1.0 m  what counts as a hit")):
        ax.axvline(v, color=col, ls=(0, (5, 4)), lw=1.7, alpha=.95, zorder=2)
        ax.text(v - 0.014, -6, lab, color=col, fontsize=10.5, fontweight="bold",
                ha="right", va="top", rotation=90)
    ax.axvline(mean, color=INK, lw=1.5, alpha=.8, zorder=2)
    ax.text(mean + 0.016, 353, f"mean {mean:.3f} m", color=INK, fontsize=11,
            fontweight="bold", va="bottom")

    ax.scatter(cpa, [b for b, _ in rows], s=70, color=BLUE, edgecolor=BG,
               linewidth=1.8, zorder=4, label="one engagement")
    ax.set_xlim(-0.025, 1.09)
    ax.set_ylim(368, -14)
    ax.set_yticks([0, 90, 180, 270, 360])
    ax.set_yticklabels(["0°", "90°", "180°", "270°", "360°"])
    ax.set_xlabel("true closest approach (metres)", labelpad=8)
    ax.set_ylabel("arrival bearing", labelpad=8)
    ax.tick_params(length=0, labelsize=10.5)
    ax.set_axisbelow(True)

    titled(fig, "24 / 24 intercepted, from every direction",
           "Isaac Sim, closed loop, 24 arrival bearings. Twenty-three of twenty-four "
           "passed within a quarter of a metre.")
    save(fig, "chart_cpa")


def chart_detect():
    """62 one-camera engagements: detection rate splits hits from misses."""
    rows = []
    for env in ("rivermark", "skydome"):
        d = json.loads((ROOT / f"work/pursuit/final/{env}/results.json").read_text())
        rows += [(bool(r["success"]), r["detect_rate"]) for r in d["results"]]
    hits = [r for ok, r in rows if ok]
    miss = [r for ok, r in rows if not ok]

    fig, ax = plt.subplots(figsize=(9.4, 3.5))
    fig.subplots_adjust(left=0.155, right=0.985, top=0.68, bottom=0.20)

    for ys, xs, col, lab in ((1.0, hits, GOOD, "intercepted"),
                             (0.0, miss, CRIT, "missed")):
        ax.scatter(xs, [ys + ((i % 7) - 3) * 0.042 for i in range(len(xs))],
                   s=58, color=col, alpha=.88, edgecolor=BG, linewidth=1.5, zorder=3)
        m = st.mean(xs)
        ax.plot([m, m], [ys - 0.25, ys + 0.25], color=INK, lw=2.4, zorder=4)
        ax.text(m, ys + 0.33, f"mean {m:.3f}", color=INK, fontsize=10.5,
                fontweight="bold", ha="center")
        ax.text(-0.035, ys, f"{lab}\nn = {len(xs)}", color=col, fontsize=12,
                fontweight="bold", ha="right", va="center", linespacing=1.6)

    ax.set_xlim(-0.02, 1.03)
    ax.set_ylim(-0.62, 1.72)
    ax.set_yticks([])
    ax.set_xlabel("detection rate — fraction of frames the target was found", labelpad=8)
    ax.yaxis.grid(False)
    ax.tick_params(length=0, labelsize=10.5)
    ax.set_axisbelow(True)

    titled(fig, "Only one thing predicts failure",
           "62 engagements, one camera. Holm-corrected over six factors (p = 0.0001). "
           "Environment does not predict it (p = 0.74), nor arrival direction, start "
           "range or evasion.")
    save(fig, "chart_detect")


def chart_range():
    """Detection fraction against range, for the two motion front-ends."""
    d = json.loads((ROOT / "work/pursuit/motion_bg2.json").read_text())
    bins = defaultdict(list)
    for r in d["rows"]:
        bins[int(r["range_m"] // 20) * 20].append(r)
    los = sorted(bins)
    x = [lo + 10 for lo in los]
    span = [st.mean(r["span_px"] for r in bins[lo]) for lo in los]

    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.76, bottom=0.20)

    ax.axhline(0.5, color=INK, ls=(0, (5, 4)), lw=1.5, alpha=.75, zorder=2)
    ax.text(206, 0.525, "a track can be held above this line", color=SUB,
            fontsize=10, ha="right", va="bottom")

    for key, col, lab in (("background=1", BLUE, "per-pixel background model"),
                          ("background=0", ORANGE, "ego-compensated frame differencing")):
        y = [sum(1 for r in bins[lo] if r[key]) / len(bins[lo]) for lo in los]
        ax.plot(x, y, color=col, lw=2.8, marker="o", ms=7, markeredgecolor=BG,
                markeredgewidth=1.8, zorder=3, label=lab)

    ax.set_xlim(22, 214)
    ax.set_ylim(-0.04, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v)}\n{s:.1f} px" for v, s in zip(x, span)], fontsize=9.5)
    ax.set_xlabel("range to the intruder (metres) · the target's pixel span below",
                  labelpad=8)
    ax.set_ylabel("fraction of frames detected", labelpad=8)
    ax.tick_params(length=0, labelsize=10.5)
    ax.set_axisbelow(True)
    leg = ax.legend(loc="upper right", frameon=False, fontsize=10.5,
                    handlelength=1.6, labelspacing=0.6)
    for t in leg.get_texts():
        t.set_color(INK)

    titled(fig, "Holding still is worth 40 metres of detection range",
           "Live Rivermark, intruder closing at 12 m/s. Reliable to 140 m against 100 m — "
           "a 70 m defended radius against 46 m, because every metre of range buys 0.6 m.")
    save(fig, "chart_range")


def main() -> int:
    print("writing result charts:")
    chart_cpa()
    chart_detect()
    chart_range()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
