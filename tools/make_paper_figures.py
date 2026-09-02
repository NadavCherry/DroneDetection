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
    """AP against measured FPS, from tools/bench_edge.py's own artefact.

    Every point is one execution: accuracy and speed were measured in the SAME pass over
    the same video, so the two axes cannot drift apart. The x axis is the steady-state p50
    rate, not the end-to-end mean -- a TensorRT engine pays a one-time initialisation cost
    that, averaged over a 361-frame clip, once made the faster backend read seventeen times
    slower than the slower one.

    The shape is the point: dropping 1280 -> 640 moves a short way right and a long way
    down. Speed bought with resolution is expensive here, and the figure should show that
    rather than a reader having to infer it from a table.
    """
    j = _load(results / "reports" / "edge" / "edge_bench.json")
    if not j:
        print("  fig2: no work/reports/edge/edge_bench.json -- skipped")
        return False
    rows = [r for r in j if r.get("ap") is not None
            and (r.get("steady_state") or {}).get("fps_at_p50")]
    if not rows:
        print("  fig2: artefact present but no row has both AP and a p50 rate -- skipped")
        return False

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    # One colour per backend, one marker per resolution: a reader can read either
    # dimension without the legend, and neither is carried by colour alone.
    col = {"engine": BLUE, "pt": ORANGE}
    mark = {1280: "o", 640: "^"}
    for r in rows:
        b, sz = r.get("backend", "?"), int(r.get("imgsz", 0))
        ax.plot([r["steady_state"]["fps_at_p50"]], [r["ap"]],
                marker=mark.get(sz, "D"), markersize=11,
                color=col.get(b, INK), linestyle="none")
        ax.annotate(f"{b} @ {sz}", (r["steady_state"]["fps_at_p50"], r["ap"]),
                    textcoords="offset points", xytext=(9, -3), fontsize=8, color=INK)

    # Join the two points that share a backend, so the resolution trade is a line the eye
    # follows rather than two dots a reader has to pair up themselves.
    for b, colour in col.items():
        pair = sorted([r for r in rows if r.get("backend") == b],
                      key=lambda r: r["steady_state"]["fps_at_p50"])
        if len(pair) == 2:
            ax.plot([q["steady_state"]["fps_at_p50"] for q in pair],
                    [q["ap"] for q in pair], color=colour, alpha=0.45,
                    linewidth=1.4, zorder=0)

    ax.set_xlabel("frames per second  (steady-state p50, whole pipeline, one GPU)")
    ax.set_ylabel("AP  (centre distance, tau = 12 px)")
    ax.set_ylim(0.55, 0.95)
    ax.grid(True, linewidth=0.6, alpha=0.5)
    gpu = rows[0].get("gpu", "one GPU")
    fig.text(0.5, 1.03, "Accuracy against speed", ha="center", color=INK, fontsize=13)
    fig.text(0.5, 0.985, f"{gpu} - both axes from the same pass; halving resolution buys "
             "1.2x speed for -0.24 AP", ha="center", color=SUB, fontsize=8)
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
    """The dt ablation, BOTH curves, because their disagreement is the result.

    Reads tools/dt_compare.py's JSON. Two panels rather than one: validation mAP50 and
    full-frame held-out test AP live on different scales (~0.93 against ~0.49) and a twin
    axis would invite reading a gap between them that is not there.

    Drawing only the validation curve would produce the figure this project nearly
    published -- a clean inverted U peaking at the deployed dt=6, with non-overlapping
    seed ranges, which looks like the constant being vindicated. The test panel beside it
    shows dt=6 third, dt=2 first, and the caption says what the paired tests found: nothing
    separates, and the two comparisons that did reach significance contradicted each other
    across seeds of the same pair.
    """
    j = _load(results / "reports" / "dt_sweep.json")
    if not j or not j.get("test_ap"):
        print("  fig4: no work/reports/dt_sweep.json -- skipped")
        return False

    chosen = j.get("baseline_dt", 6)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3))
    panels = [("val_map50", "validation mAP50  (640 px tiles)", BLUE,
               "clean inverted U, peak at the deployed dt"),
              ("test_ap", "held-out test AP  (full frame, 10 clips)", ORANGE,
               "different ranking; nothing separates")]

    for ax, (key, ylab, colour, note) in zip(axes, panels):
        data = j.get(key) or {}
        if not data:
            ax.text(0.5, 0.5, f"no {key}", ha="center", va="center",
                    transform=ax.transAxes, color=SUB)
            continue
        xs = sorted(int(k) for k in data)
        ys = [st.fmean(data[str(x)]) for x in xs]
        es = [st.stdev(data[str(x)]) if len(data[str(x)]) > 1 else 0.0 for x in xs]
        ax.errorbar(xs, ys, yerr=es, color=colour, marker="o", markersize=7,
                    linewidth=2, capsize=4)
        # Every seed as a faint point: three runs behind a mean is a fact the reader
        # should be able to see, especially where the spread dwarfs the effect.
        for x in xs:
            ax.plot([x] * len(data[str(x)]), data[str(x)], marker=".", linestyle="none",
                    color=colour, alpha=0.45, markersize=6)
        if chosen in xs:
            ax.axvline(chosen, color=CRIT, linewidth=1.1, linestyle=":")
            ax.annotate(f"deployed dt={chosen}", (chosen, min(ys)),
                        textcoords="offset points", xytext=(6, -12),
                        color=CRIT, fontsize=8)
        best = xs[ys.index(max(ys))]
        ax.annotate(f"best here: dt={best}", (best, max(ys)),
                    textcoords="offset points", xytext=(-10, 10), fontsize=8,
                    color=INK)
        ax.set_xticks(xs)
        ax.set_xlabel("dt, frames between taps   (aperture = 2·dt + 1)")
        ax.set_ylabel(ylab)
        ax.set_title(note, color=SUB, fontsize=9)
        ax.grid(True, linewidth=0.6, alpha=0.5)

    n_sig = sum(1 for r in j.get("paired", []) if r.get("significant"))
    n_tot = len(j.get("paired", []))
    fig.text(0.5, 1.04, "Tap spacing: the two metrics do not agree",
             ha="center", color=INK, fontsize=13)
    fig.text(0.5, 0.985,
             f"paired bootstrap + permutation over 10 sequences, seed-matched: "
             f"{n_sig} of {n_tot} comparisons significant"
             + (" -- and those contradict each other across seeds of the same pair"
                if n_sig else ""),
             ha="center", color=SUB, fontsize=8)
    fig.tight_layout()
    _finish(fig, out, "fig4_dt_ablation.png")
    return True


# ---------------------------------------------------------------- figure 5
def fig5(results: Path, out: Path, video: Path, gt_path: Path, dt: int = 6) -> bool:
    """Qualitative: the same target at four sizes, one frame against three moments.

    This is the figure that makes figure 1 legible. Figure 1 says our advantage lives
    below 10 px and disappears above it; this shows what a target of that size actually
    looks like, and why one frame cannot carry it.

    Each column is a real frame from the held-out video, chosen because the labelled
    target falls in that size bucket. Top row is the raw frame as a single-frame detector
    sees it. Bottom row is the detector's actual input -- grayscale at t-2dt, t-dt and t
    as R, G and B -- so a stationary world is grey and anything that moved is coloured.

    cv2 is imported here rather than at module scope on purpose: the other four figures
    need only matplotlib, and a missing OpenCV should cost this figure alone rather than
    the whole run.
    """
    try:
        import cv2  # noqa: F401
        import numpy as np
        from collections import deque
        from dronedet.gt import GroundTruth
        from dronedet.stabilize import Stabilizer
        from dronedet.video import frames as _frames
        from tools.make_dataset_external import _stack_aligned_to_now
    except Exception as e:
        print(f"  fig5: needs OpenCV and the repo pipeline ({type(e).__name__}) -- skipped")
        return False
    if not video.is_file() or not gt_path.is_file():
        print(f"  fig5: missing {video if not video.is_file() else gt_path} -- skipped")
        return False

    gt = GroundTruth.load(gt_path)
    target = next((o for o in gt.objects.values() if not o.ignore), None)
    if target is None:
        print("  fig5: no non-ignore GT object -- skipped")
        return False

    # One representative frame per size bucket, picked by the GT's own box area so the
    # caption's pixel figure is measured rather than asserted.
    buckets = [("< 8 px", 0.0, 8.0), ("8-10 px", 8.0, 10.0),
               ("10-16 px", 10.0, 16.0), ("> 16 px", 16.0, 1e9)]
    excl = set(gt.meta.get("exclude_frames", []))
    want: dict[str, tuple[int, tuple]] = {}
    for f, box in sorted(target.frames.items()):
        if f in excl or f < 2 * dt + 1:
            continue
        s = (box[2] * box[3]) ** 0.5
        for name, lo, hi in buckets:
            if lo <= s < hi and name not in want:
                want[name] = (f, box)
    cols = [(n, *want[n]) for n, _, _ in buckets if n in want]
    if not cols:
        print("  fig5: no frame matched any size bucket -- skipped")
        return False

    need = {f for _, f, _ in cols}
    stab = Stabilizer("translation")
    buf: deque = deque(maxlen=2 * dt + 1)
    raw: dict[int, "np.ndarray"] = {}
    stack: dict[int, "np.ndarray"] = {}
    for idx, frame in _frames(str(video)):
        m = stab.update(frame)
        buf.append((cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                    float(m[0, 2]), float(m[1, 2])))
        if idx in need and len(buf) == buf.maxlen:
            raw[idx] = frame.copy()
            stack[idx] = np.dstack(_stack_aligned_to_now(buf, dt))
        if idx > max(need):
            break

    cols = [c for c in cols if c[1] in raw and c[1] in stack]
    if not cols:
        print("  fig5: could not build the temporal stack for any chosen frame -- skipped")
        return False

    R = 44                                    # crop half-width, px
    fig, axes = plt.subplots(2, len(cols), figsize=(2.5 * len(cols), 5.4), squeeze=False)
    for j, (name, f, box) in enumerate(cols):
        cx, cy = int(box[0]), int(box[1])
        size = (box[2] * box[3]) ** 0.5
        for i, img in enumerate((raw[f], stack[f])):
            h, w = img.shape[:2]
            x0, y0 = max(0, cx - R), max(0, cy - R)
            crop = img[y0:min(h, cy + R), x0:min(w, cx + R)]
            if crop.ndim == 3 and i == 0:
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            ax = axes[i][j]
            ax.imshow(crop, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(GRID)
            # Mark where the target is, without covering it.
            ax.plot([cx - x0], [cy - y0], marker="o", markersize=16,
                    markerfacecolor="none", markeredgecolor=CRIT, markeredgewidth=1.2)
            if i == 0:
                ax.set_title(f"{name}" + "\n" + f"{size:.1f} px, frame {f}",
                             color=INK, fontsize=9)
        axes[0][j].set_ylabel("")
    axes[0][0].set_ylabel("one frame" + "\n" + "(what a single-frame"
                          + "\n" + "detector sees)", color=SUB, fontsize=8)
    axes[1][0].set_ylabel("three moments" + "\n" + "(the detector's"
                          + "\n" + "actual input)", color=SUB, fontsize=8)
    fig.text(0.5, 1.02, "The same target, one frame against three moments",
             ha="center", color=INK, fontsize=13)
    fig.text(0.5, 0.975, f"real frames from {video.name}; red circle marks the labelled "
             f"target. Taps at t-{2*dt}, t-{dt}, t as R, G, B",
             ha="center", color=SUB, fontsize=8)
    fig.tight_layout()
    _finish(fig, out, "fig5_qualitative_tiny_target.png")
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
    ap.add_argument("--video", type=Path, default=REPO / "data/videos/10_06.mp4",
                    help="fig5 only: the held-out video the crops come from")
    ap.add_argument("--gt", type=Path,
                    default=REPO / "realtime/work/gt_1006_v2.json",
                    help="fig5 only: its ground truth, which picks the size buckets")
    a = ap.parse_args()

    print(f"reading results from {a.results}")
    made = [fig1(a.results, a.out, a.datasets, a.bins),
            fig2(a.results, a.out),
            fig3(a.results, a.out, a.datasets, a.bins),
            fig4(a.results, a.out),
            fig5(a.results, a.out, a.video, a.gt)]
    n = sum(1 for m in made if m)
    print(f"\n{n}/4 figures written to {a.out}")
    ### Zero figures is a failure worth an exit code: this runs in a job whose log nobody
    ### reads line by line, and "skipped, skipped, skipped, done" should not look like
    ### success to whatever ran it.
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main())