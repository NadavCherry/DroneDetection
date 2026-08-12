#!/usr/bin/env python3
"""Did we stop it, or did the building get hit? The city-defence scorecard.

Every other report in this package scores an interception. This one scores a
*defence*, and the difference is a column: an engagement can be lost by missing
and it can be lost by arriving late, and only the second one has a building in
it. So the headline here is a pair -- intercepted against struck -- and every
supporting number exists to explain which of the two happened and why.

    python -m pursuit.tools.city_report --search work/pursuit/city --out METRICS.md

What it will not do is average those two failures together. "83 percent
success" is a sentence about an interceptor; "20 of 24 intruders stopped, 4
buildings hit, and here is the arithmetic that says which four were reachable"
is a sentence about a defence.
"""
from __future__ import annotations

import argparse
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pursuit.tools.report import load  # noqa: E402


from dronedet.stats import wilson  # noqa: E402  (one implementation, tested there)


def _f(v: Optional[float], nd: int = 2, dash: str = "—") -> str:
    return dash if v is None else f"{v:.{nd}f}"


def _mean(xs: Sequence[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return st.fmean(xs) if xs else None


#: Where the ring was last flown with a real detector in the loop. The city
#: campaign itself is scored with the oracle sensor, so its loop time says
#: nothing about the pipeline rate and this run is what the report quotes
#: instead. Anything that changes the ring's cost invalidates it -- re-fly it.
PIPELINE_RUN = "work/pursuit/city_pipe"


def pipeline_rate(search: Path) -> Optional[str]:
    """The ring's measured loop time, as a sentence, or ``None``.

    Read off a run that actually carried the detector: a run whose appearance
    stage costs less than a millisecond has no detector in it, and quoting its
    loop as a frame rate is how a report ends up claiming a rate nobody
    measured. If no such run is on disk this returns ``None`` and the caller
    says so, which is the honest output when the campaign did not time itself.
    """
    stages = []
    for run in load(search):
        for r in run["payload"]["results"]:
            sm = r.get("stage_ms") or {}
            if (sm.get("detect_ms") or 0.0) >= 1.0:
                stages.append(sm)
    if not stages:
        return None
    det = _mean([s.get("detect_ms") for s in stages])
    mot = _mean([s.get("motion_ms") for s in stages])
    trk = _mean([s.get("track_ms") for s in stages])
    per = [s["perception_ms"] for s in stages if s.get("perception_ms")]
    fps = [s["perception_fps"] for s in stages if s.get("perception_fps")]
    if not per or not fps:
        return None
    return (
        f"The pipeline rate is measured separately, with the real sensor on live "
        f"Rivermark (`{PIPELINE_RUN}`, {len(stages)} engagement"
        f"{'' if len(stages) == 1 else 's'}): appearance detector "
        f"**{det:.1f} ms** on a 640 px crop, motion detector **{mot:.1f} ms** "
        f"across four 2048x704 images (threaded), tracker **{trk:.1f} ms** — "
        f"**{st.fmean(per):.0f} ms of perception, {st.fmean(fps):.1f} FPS** "
        f"({min(fps):.1f}–{max(fps):.1f} across the run). Aiming the network at a "
        f"crop made the appearance stage roughly eight times cheaper than the "
        f"nose camera's whole-frame pass (130.7 ms, `work/pursuit/final/METRICS.md`); "
        f"what is left is almost entirely the classical motion stage, run over "
        f"four wide images of a cluttered city."
    )


def collect(search: Path) -> list:
    rows = []
    for run in load(search):
        for r in run["payload"]["results"]:
            cfg = r.get("config") or {}
            if not cfg.get("defend_xy"):
                continue
            sm = r.get("stage_ms") or {}
            rows.append({
                "name": r.get("name", "?"),
                "building": cfg.get("defend_label") or "?",
                "bearing": cfg.get("entry") or "?",
                "hit": bool(r.get("success")),
                "struck": bool(r.get("struck_asset")),
                "outcome": r.get("outcome", "?"),
                "cpa": r.get("pass_cpa_m"),
                "t": r.get("time_to_intercept_s"),
                "acq": r.get("acquire_time_s"),
                "margin": r.get("strike_margin_s"),
                "min_asset": r.get("min_asset_range_m"),
                "det": r.get("detect_rate"),
                "trk": r.get("track_rate"),
                "off": r.get("offtarget_rate"),
                "asset_range": math.hypot(*cfg["defend_xy"]),
                "r0": cfg.get("start_range_m"),
                "v_i": cfg.get("transit_speed") or cfg.get("evader_speed"),
                "adv": cfg.get("speed_advantage"),
                "det_ms": sm.get("detect_ms"), "motion_ms": sm.get("motion_ms"),
                "trk_ms": sm.get("track_ms"), "gui_ms": sm.get("guidance_ms"),
                "p95": sm.get("pipeline_p95_ms"),
            })
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--search", default="work/pursuit/city")
    ap.add_argument("--out", default="")
    ap.add_argument("--title", default="City defence — Rivermark, four-camera ring")
    ap.add_argument("--pipeline-run", default=PIPELINE_RUN,
                    help="run to read the ring's loop time from, when this "
                         "campaign was flown on the oracle sensor")
    ap.add_argument("--readme", action="store_true",
                    help="also emit the compact block the top-level "
                         "README carries, so the headline is generated "
                         "rather than transcribed")
    a = ap.parse_args(argv)

    rows = collect(ROOT / a.search)
    if not rows:
        print(f"no defence results under {ROOT / a.search}")
        return 1

    L = []

    def out(s=""):
        L.append(s)
        print(s)

    n = len(rows)
    hits = [r for r in rows if r["hit"]]
    struck = [r for r in rows if r["struck"]]
    other = [r for r in rows if not r["hit"] and not r["struck"]]
    p, lo, hi = wilson(len(hits), n)

    out(f"# {a.title}\n")
    out(f"{n} engagements. The interceptor rises over the middle of the town and "
        f"holds; an intruder arrives from a bearing drawn from the whole circle "
        f"and flies at the nearest building. It is 1.5x slower than the "
        f"interceptor and it does not break off.\n")

    out("## Outcome\n")
    out("| outcome | n | share |")
    out("|---|---|---|")
    out(f"| **intruder intercepted** | **{len(hits)}** | "
        f"**{100.0 * len(hits) / n:.1f}%** |")
    out(f"| **building struck** | **{len(struck)}** | "
        f"{100.0 * len(struck) / n:.1f}% |")
    out(f"| neither (timed out / never acquired) | {len(other)} | "
        f"{100.0 * len(other) / n:.1f}% |")
    out("")
    out(f"Interception rate **{len(hits)}/{n} = {100 * p:.1f}%**, "
        f"95% CI [{100 * lo:.1f}%, {100 * hi:.1f}%] (Wilson). "
        f"**{len(struck)} building{'' if len(struck) == 1 else 's'} hit.**\n")

    if struck:
        out("Struck, with the reason each one was or was not reachable "
            "— the interceptor needs to acquire beyond `asset x (1 + v_i/v_c)` "
            "to beat a head-on run:\n")
        out("| scenario | building | asset range | needed acquisition | "
            "closest the intruder got to us | detection rate |")
        out("|---|---|---|---|---|---|")
        for r in sorted(struck, key=lambda r: -r["asset_range"]):
            need = r["asset_range"] * (1.0 + 1.0 / (r["adv"] or 1.5))
            out(f"| {r['name']} | {r['building']} | {r['asset_range']:.0f} m | "
                f"{need:.0f} m | {_f(r['min_asset'], 0)} m | {_f(r['det'])} |")
        out("")

    # -- per building --------------------------------------------------------
    by_b = defaultdict(list)
    for r in rows:
        by_b[r["building"]].append(r)
    out("## By building\n")
    out("| building | asset range | engagements | intercepted | struck | "
        "median margin | mean CPA |")
    out("|---|---|---|---|---|---|---|")
    for b, rs in sorted(by_b.items(), key=lambda kv: st.fmean(
            [r["asset_range"] for r in kv[1]])):
        mg = [r["margin"] for r in rs if r["margin"] is not None]
        cp = [r["cpa"] for r in rs if r["cpa"] is not None]
        out(f"| {b} | {st.fmean([r['asset_range'] for r in rs]):.0f} m | "
            f"{len(rs)} | {sum(1 for r in rs if r['hit'])} | "
            f"{sum(1 for r in rs if r['struck'])} | "
            f"{_f(st.median(mg) if mg else None)} s | "
            f"{_f(st.fmean(cp) if cp else None)} m |")
    out("")

    # -- quality of the intercepts ------------------------------------------
    cp = sorted(r["cpa"] for r in hits if r["cpa"] is not None)
    mg = sorted(r["margin"] for r in hits if r["margin"] is not None)
    tt = sorted(r["t"] for r in hits if r["t"] is not None)
    aq = sorted(r["acq"] for r in rows if r["acq"] is not None)
    if cp:
        out("## The intercepts themselves\n")
        out("| metric | value |")
        out("|---|---|")
        out(f"| mean true closest approach | **{st.fmean(cp):.3f} m** |")
        out(f"| median / p90 | {st.median(cp):.3f} / "
            f"{cp[min(len(cp) - 1, int(0.9 * len(cp)))]:.3f} m |")
        out(f"| inside 0.5 m | {100 * sum(1 for v in cp if v <= 0.5) / len(cp):.0f}% |")
        if mg:
            out(f"| median time to spare before the strike | "
                f"**{st.median(mg):.2f} s** |")
            out(f"| worst margin | {min(mg):.2f} s |")
        if tt:
            out(f"| median time to intercept | {st.median(tt):.2f} s |")
        if aq:
            out(f"| median time to acquire | **{st.median(aq):.2f} s** "
                f"(p90 {aq[min(len(aq) - 1, int(0.9 * len(aq)))]:.2f} s) |")
        out(f"| mean detection rate (of frames the target was in a camera) | "
            f"{_f(_mean([r['det'] for r in rows]))} |")
        out(f"| mean track rate | {_f(_mean([r['trk'] for r in rows]))} |")
        out(f"| **time locked on something that was not the drone** | "
            f"**{_f(_mean([r['off'] for r in rows]))}** of tracked frames |")
        out("")

    # -- acquisition is the whole game --------------------------------------
    dh = [r["det"] for r in hits if r["det"] is not None]
    dm = [r["det"] for r in rows if not r["hit"] and r["det"] is not None]
    if dh and dm:
        out("## Perception versus the clock\n")
        out(f"- detection rate on **saves**: {st.fmean(dh):.3f} (n={len(dh)})")
        out(f"- detection rate on **losses**: {st.fmean(dm):.3f} (n={len(dm)})")
        out(f"- median acquisition {_f(st.median(aq) if aq else None)} s — with "
            f"a ring there is no search to do, so this is the detector's range "
            f"and nothing else.\n")

    # -- timing --------------------------------------------------------------
    stages = [("appearance detector", "det_ms"), ("motion detector", "motion_ms"),
              ("tracker", "trk_ms"), ("guidance", "gui_ms")]
    vals = {k: _mean([r[k] for r in rows]) for _lbl, k in stages}
    loop = sum(v for v in vals.values() if v)
    if loop > 0 and (vals.get("det_ms") or 0.0) < 1.0:
        # An oracle run has no detector in it, so its loop time is a statement
        # about the tracker and nothing else. Printing "5000 FPS" next to a
        # closure result would be the most misreadable number in the document.
        out("## Frame rate\n")
        measured = pipeline_rate(ROOT / a.pipeline_run)
        out("*Not measured here.* This run scores **closure**, and its sensor is "
            "the simulator's own bounding box -- there is no detector in the "
            "loop, so the loop time below is the tracker and the guidance law "
            "alone. " + (measured if measured else
                         f"No run under `{a.pipeline_run}` carries a real "
                         f"detector, so the ring's pipeline rate is not "
                         f"measured anywhere on this disk and no figure is "
                         f"quoted here.") + "\n")
    if loop > 0:
        out("### stage breakdown\n" if (vals.get("det_ms") or 0.0) < 1.0
            else "## Frame rate\n")
        out("| stage | ms | share |")
        out("|---|---|---|")
        for lbl, k in stages:
            v = vals.get(k) or 0.0
            out(f"| {lbl} | {v:.2f} | {100.0 * v / loop:.0f}% |")
        out(f"| **total** | **{loop:.2f}** | |")
        p95 = _mean([r["p95"] for r in rows])
        out("")
        out(f"- **{1000.0 / loop:.1f} FPS** mean"
            + (f", **{1000.0 / p95:.1f} FPS** at p95" if p95 else ""))
        out(f"- 20 Hz control budget is 50 ms — "
            f"{'**met**' if loop <= 50 else '**not met**'} ({loop:.1f} ms)\n")

    out("## Every engagement\n")
    out("| scenario | building | outcome | acquire | t | CPA | margin | det |")
    out("|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: r["name"]):
        mark = "**HIT**" if r["hit"] else ("STRUCK" if r["struck"] else r["outcome"])
        out(f"| {r['name']} | {r['building']} | {mark} | {_f(r['acq'])} s | "
            f"{_f(r['t'])} s | {_f(r['cpa'], 3)} m | {_f(r['margin'])} s | "
            f"{_f(r['det'])} |")
    out("")

    if a.readme:
        # The compact block the top-level README carries, generated rather than
        # transcribed -- a hand-copied headline is a number with no provenance
        # and the first thing to go stale.
        block = [
            "| outcome | n | share |",
            "|---|---|---|",
            f"| **intruder intercepted** | **{len(hits)}** | "
            f"**{100.0 * len(hits) / n:.1f}%** |",
            f"| **building struck** | **{len(struck)}** | "
            f"{100.0 * len(struck) / n:.1f}% |",
        ]
        if other:
            block.append(f"| neither | {len(other)} | "
                         f"{100.0 * len(other) / n:.1f}% |")
        block += [
            "",
            f"**{len(hits)} of {n} intruders stopped** "
            f"({100 * p:.1f} %, 95 % CI [{100 * lo:.1f} %, {100 * hi:.1f} %] "
            f"Wilson) — **{len(struck)} building"
            f"{'' if len(struck) == 1 else 's'} hit**. "
            + (f"Mean true closest approach **{st.fmean(cp):.3f} m**, "
               f"median **{st.median(mg):.2f} s** of margin before the strike, "
               if cp and mg else "")
            + f"median acquisition **{_f(st.median(aq) if aq else None)} s** — "
              f"with 360 degree coverage there is no search to do.",
        ]
        print("\n" + "\n".join(block))
        Path(ROOT / "work/pursuit/city/README_BLOCK.md").write_text(
            "\n".join(block), encoding="utf-8")

    if a.out:
        p = Path(a.out)
        if not p.is_absolute():
            p = ROOT / p
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(L), encoding="utf-8")
        print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
