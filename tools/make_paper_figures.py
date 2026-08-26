#!/usr/bin/env python3
"""The publication figures, each generated from a saved result file and nothing else.

Every figure here reads a JSON artefact produced by an evaluation job. None of them
contain a number typed by hand, so a figure cannot drift from the run that produced it --
and re-running this after a re-evaluation is the whole update procedure.

    fig1_accuracy_vs_size.png    AP against target size, per dataset. THE figure: it is
                                 the one that turns "we lose overall but win on small
                                 targets" from a claim into a curve with error bars.
    fig2_accuracy_vs_speed.png   AP against measured FPS, both arms on one GPU.
    fig3_single_vs_temporal.png  what the three-frame stack actually buys, by size.
    fig4_dt_ablation.png         AP against tap spacing dt.

Inputs, all optional -- a figure whose input is missing is SKIPPED with a message rather
than drawn from defaults. A plausible-looking figure built from absent data is worse than
no figure, because nothing downstream can tell it was invented.

    work/reports/size_curve/{dataset}_{bins}.json   figs 1 and 3
    work/reports/bench/*.json                       fig 2
    work/reports/ablation/dt_sweep.json             fig 4

    PYTHONPATH=. python tools/make_paper_figures.py --results work --out docs/media/paper

Palette and dark canvas match tools/make_result_charts.py so the README reads as one set.
The blue/orange pair is CVD-safe; every series is also named in a legend, so colour never
carries the identity on its own.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

BG = "#0d1117"
INK, SUB, GRID = "#e6edf3", "#9198a1", "#30363d"
BLUE, ORANGE, GREEN = "#58a6ff", "#f0883e", "#3fb950"
CRIT = "#f85149"

#: Stable colour and marker per arm. Marker as well as colour, because these figures are
#: read on LinkedIn at thumbnail size and in print.
ARM_STYLE = {
    "ours": (BLUE, "o", "SpeckLock (ours, temporal)"),
    "yolomg": (ORANGE, "s", "YOLOMG (competitor, retrained by us)"),
    "ours-single": (GREEN, "^", "ours, single-frame control"),
}

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "savefig.facecolor": BG, "savefig.edgecolor": BG,
    "text.color": INK, "axes.labelcolor": SUB, "axes.edgecolor": GRID,
    "xtick.color": SUB, "ytick.color": SUB, "grid.color": GRID,
    "font.size": 10, "axes.titlesize": 11, "legend.frameon": False,
})


def _load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _finish(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------- figure 1
def fig1(results: Path, out: Path, datasets: list[str], bins: str = "mission") -> bool:
    """AP against target size, one panel per dataset.

    Shaded band is the 95 % bootstrap interval over SEQUENCES. Bins whose GT count is
    small are drawn hollow and annotated, because an AP over a handful of boxes is noise
    and a reader scanning the shape of a curve will otherwise read it as signal.
    """
    found = [(d, _load(results / "reports" / "size_curve" / f"{d}_{bins}.json"))
             for d in datasets]
    found = [(d, j) for d, j in found if j]
    if not found:
        print("  fig1: no size_curve JSON found -- skipped")
        return False

    fig, axes = plt.subplots(1, len(found), figsize=(5.2 * len(found), 4.2),
                             squeeze=False)
    for ax, (ds, j) in zip(axes[0], found):
        arms = j["arms"]
        order = [k for k in ("ours", "yolomg", "ours-single") if k in arms]
        labels: list[str] = []
        for arm in order:
            cells = arms[arm]["bins"]
            names = [b for b in cells]
            if not labels:
                labels = names
            xs = list(range(len(labels)))
            ys = [cells[b]["mean"] if b in cells else float("nan") for b in labels]
            lo = [cells[b]["ci95"][0] if b in cells else float("nan") for b in labels]
            hi = [cells[b]["ci95"][1] if b in cells else float("nan") for b in labels]
            colour, marker, legend = ARM_STYLE.get(arm, (INK, "o", arm))
            ax.fill_between(xs, lo, hi, color=colour, alpha=0.13, linewidth=0)
            ax.plot(xs, ys, color=colour, marker=marker, markersize=6,
                    linewidth=2, label=legend)
            # Hollow marker wherever the bin is underpowered.
            for x, b in zip(xs, labels):
                if b in cells and cells[b].get("underpowered"):
                    ax.plot([x], [cells[b]["mean"]], marker=marker, markersize=6,
                            markerfacecolor=BG, markeredgecolor=colour, linewidth=0)

        n_row = [str(arms[order[0]]["bins"][b]["n_gt"]) if b in arms[order[0]]["bins"]
                 else "-" for b in labels]
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([f"{b}\nn={n}" for b, n in zip(labels, n_row)], fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(True, axis="y", linewidth=0.6, alpha=0.5)
        ax.set_title(ds, color=INK, pad=8)
        ax.set_xlabel("target size, sqrt(area)", labelpad=6)
    axes[0][0].set_ylabel("AP @ IoU 0.5")
    axes[0][0].legend(loc="upper left", fontsize=8)
    fig.text(0.5, 1.04, "Accuracy against target size", ha="center",
             color=INK, fontsize=13)
    fig.text(0.5, 1.005,
             "band = 95 % bootstrap CI over sequences; hollow marker = underpowered bin",
             ha="center", color=SUB, fontsize=8)
    _finish(fig, out, "fig1_accuracy_vs_size.png")
    return True


# ---------------------------------------------------------------- figure 2
def fig2(results: Path, out: Path) -> bool:
    """AP against measured FPS. Both axes measured by us on one GPU."""
    bench_dir = results / "reports" / "bench"
    rows = []
    for p in sorted(bench_dir.glob("*.json")) if bench_dir.is_dir() else []:
        j = _load(p)
        if j and j.get("fps_at_p50"):
            rows.append(j)
    if not rows:
        print("  fig2: no benchmark JSON found -- skipped")
        return False

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for j in rows:
        arm = j.get("arm", "?")
        colour, marker, legend = ARM_STYLE.get(arm, (INK, "D", arm))
        ap = j.get("ap")
        label = j.get("label") or legend
        if j.get("engine") in (".engine", ".onnx"):
            label += f" [{j['engine'].lstrip('.')}]"
        if ap is None:
            # Speed measured but no AP attached: draw it on the axis floor and say so,
            # rather than silently dropping a measurement that cost GPU time.
            ax.plot([j["fps_at_p50"]], [0.02], marker=marker, markersize=9,
                    color=colour, markerfacecolor=BG)
            ax.annotate(f"{label}\n(AP not attached)", (j["fps_at_p50"], 0.02),
                        textcoords="offset points", xytext=(6, 6),
                        fontsize=7, color=SUB)
            continue
        ax.plot([j["fps_at_p50"]], [ap], marker=marker, markersize=10, color=colour)
        ax.annotate(label, (j["fps_at_p50"], ap), textcoords="offset points",
                    xytext=(8, 4), fontsize=8, color=INK)

    ax.set_xlabel("frames per second (p50, whole pipeline, one GPU)")
    ax.set_ylabel("AP @ IoU 0.5")
    ax.set_ylim(0, 1)
    ax.grid(True, linewidth=0.6, alpha=0.5)
    fig.text(0.5, 1.02, "Accuracy against speed", ha="center", color=INK, fontsize=13)
    fig.text(0.5, 0.98, "both arms timed in one process on one GPU, "
             "pre- and post-processing included", ha="center", color=SUB, fontsize=8)
    _finish(fig, out, "fig2_accuracy_vs_speed.png")
    return True


# ---------------------------------------------------------------- figure 3
def fig3(results: Path, out: Path, datasets: list[str], bins: str = "mission") -> bool:
    """What the temporal stack buys over the single-frame control, by target size.

    Plotted as a DIFFERENCE with a zero line, not as two curves. The two curves version
    of this figure was unreadable: both arms sit within a few points of each other over
    most of the range, and the eye cannot integrate the gap across a log-ish x axis.
    """
    found = [(d, _load(results / "reports" / "size_curve" / f"{d}_{bins}.json"))
             for d in datasets]
    found = [(d, j) for d, j in found
             if j and "ours" in j.get("arms", {}) and "ours-single" in j.get("arms", {})]
    if not found:
        print("  fig3: no size_curve JSON with both temporal and single-frame -- skipped")
        return False

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    palette = [BLUE, ORANGE, GREEN]
    labels: list[str] = []
    for i, (ds, j) in enumerate(found):
        t = j["arms"]["ours"]["bins"]
        s = j["arms"]["ours-single"]["bins"]
        keys = [b for b in t if b in s]
        if not labels:
            labels = keys
        xs = list(range(len(labels)))
        ys = [t[b]["mean"] - s[b]["mean"] if b in t and b in s else float("nan")
              for b in labels]
        ax.plot(xs, ys, marker="o", markersize=6, linewidth=2,
                color=palette[i % len(palette)], label=ds)

    ax.axhline(0, color=SUB, linewidth=1, linestyle="--")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("target size, sqrt(area)", labelpad=6)
    ax.set_ylabel("AP(temporal) - AP(single-frame)")
    ax.grid(True, axis="y", linewidth=0.6, alpha=0.5)
    ax.legend(fontsize=8)
    fig.text(0.5, 1.03, "What the three-frame stack buys, by target size",
             ha="center", color=INK, fontsize=13)
    fig.text(0.5, 0.99, "above zero = temporal helps; below = the single-frame "
             "control is better", ha="center", color=SUB, fontsize=8)
    _finish(fig, out, "fig3_single_vs_temporal.png")
    return True


# ---------------------------------------------------------------- figure 4
def fig4(results: Path, out: Path) -> bool:
    """AP against tap spacing dt, with the deployed choice marked.

    Expects {"dataset": ..., "points": [{"dt":int,"ap_mean":float,"ap_std":float,
    "seeds":[...], "fps":float|null}, ...], "chosen": int}.
    """
    j = _load(results / "reports" / "ablation" / "dt_sweep.json")
    if not j or not j.get("points"):
        print("  fig4: no dt_sweep.json found -- skipped")
        return False

    pts = sorted(j["points"], key=lambda p: p["dt"])
    xs = [p["dt"] for p in pts]
    ys = [p["ap_mean"] for p in pts]
    es = [p.get("ap_std", 0.0) for p in pts]

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.errorbar(xs, ys, yerr=es, color=BLUE, marker="o", markersize=7,
                linewidth=2, capsize=4, label="AP @ IoU 0.5")
    chosen = j.get("chosen")
    if chosen in xs:
        ax.axvline(chosen, color=CRIT, linewidth=1.2, linestyle=":")
        ax.annotate(f"deployed: dt={chosen}", (chosen, min(ys)),
                    textcoords="offset points", xytext=(6, -2),
                    color=CRIT, fontsize=8)
    ax.set_xticks(xs)
    ax.set_xlabel("dt, frames between taps (aperture = 2*dt + 1)")
    ax.set_ylabel("AP @ IoU 0.5")
    ax.grid(True, linewidth=0.6, alpha=0.5)
    ax.legend(fontsize=8)
    fig.text(0.5, 1.02, f"Tap spacing ablation -- {j.get('dataset','')}",
             ha="center", color=INK, fontsize=13)
    _finish(fig, out, "fig4_dt_ablation.png")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=REPO / "work",
                    help="root holding reports/ (default: work/)")
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "media" / "paper")
    ap.add_argument("--datasets", nargs="*",
                    default=["nps", "ardmav", "local_ft"])
    ap.add_argument("--bins", default="mission")
    a = ap.parse_args()

    print(f"reading results from {a.results}")
    made = [fig1(a.results, a.out, a.datasets, a.bins),
            fig2(a.results, a.out),
            fig3(a.results, a.out, a.datasets, a.bins),
            fig4(a.results, a.out)]
    n = sum(1 for m in made if m)
    print(f"\n{n}/4 figures written to {a.out}")
    ### Zero figures is a failure worth an exit code: this runs in a job whose log nobody
    ### reads line by line, and "skipped, skipped, skipped, done" should not look like
    ### success to whatever ran it.
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main())
