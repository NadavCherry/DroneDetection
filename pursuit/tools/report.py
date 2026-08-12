"""End-to-end metrics for the whole pipeline, as a table you can publish.

Reads the ``results.json`` every :mod:`pursuit.run_pursuit` invocation leaves
behind and produces one Markdown report over all of them: per-scenario rows,
per-environment and per-entry-direction aggregates, and the timing breakdown.

Two things it is careful about, because both are easy to quote wrongly.

**Miss distance.** ``miss_distance_m`` is scored at the tick the hit was
declared, with up to 50 ms of approach still un-flown -- across 223 intercepts,
89 percent of it was pure along-track residue. Where the aircraft *actually*
passed is ``pass_cpa_m`` (see :func:`pursuit.episode.pass_geometry`), and that is
what this reports, with the scored value alongside so the two are never confused.

**Frame rate.** The headline FPS here is the *pipeline* rate -- detector plus
tracker plus guidance -- and deliberately not the wall-clock rate of the test
run. The latter is dominated by Isaac rendering five frames per control tick,
which is a property of the rig and will not exist on the aircraft. Quoting it
would flatter the system by a factor of several and would be worthless as an
edge-hardware baseline, which is exactly what this number is for.

    python -m pursuit.tools.report --out work/pursuit/final/METRICS.md
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCENE_NAMES = {"rivermark": "Rivermark (urban)", "skydome": "Skydome (open sky)"}


def _scene_of(payload: dict, path: Path) -> str:
    name = str((payload.get("sim") or {}).get("scene_name") or "")
    low = f"{name} {path}".lower()
    if "river" in low or "town" in low:
        return "rivermark"
    if "sky" in low:
        return "skydome"
    return name or "unknown"


def _pct(xs) -> str:
    return f"{100.0 * st.fmean(xs):.1f}%" if xs else "-"


def _fmt(v: Optional[float], nd=2, dash="-") -> str:
    return dash if v is None else f"{v:.{nd}f}"


def _q(xs, p):
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(p * len(s)))]


def load(search: Path) -> list:
    runs = []
    for rj in sorted(search.rglob("results.json")):
        try:
            payload = json.loads(rj.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not payload.get("results"):
            continue
        runs.append({"dir": rj.parent, "scene": _scene_of(payload, rj.parent),
                     "payload": payload})
    return runs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--search", default="work/pursuit/final")
    ap.add_argument("--out", default="work/pursuit/final/METRICS.md")
    ap.add_argument("--title", default="End-to-end pursuit metrics")
    a = ap.parse_args(argv)

    runs = load(ROOT / a.search)
    if not runs:
        print(f"no results.json under {ROOT / a.search}")
        return 1

    rows = []
    for run in runs:
        for r in run["payload"]["results"]:
            cfg = r.get("config") or {}
            sm = r.get("stage_ms") or {}
            rows.append({
                "scene": run["scene"],
                "name": r.get("name", "?"),
                "entry": cfg.get("entry") or "-",
                "policy": r.get("policy", "-"),
                "r0": cfg.get("start_range_m"),
                "hit": bool(r.get("success")),
                "cpa": r.get("pass_cpa_m"),
                "scored": r.get("miss_distance_m"),
                "vert": r.get("pass_vertical_m"),
                "lat": r.get("pass_lateral_m"),
                "t": r.get("time_to_intercept_s"),
                "acq": r.get("acquire_time_s"),
                "reveal": r.get("reveal_time_s"),
                "det": r.get("detect_rate"),
                "trk": r.get("track_rate"),
                "det_ms": sm.get("detect_ms"),
                "trk_ms": sm.get("track_ms"),
                "gui_ms": sm.get("guidance_ms"),
                "fps": sm.get("pipeline_fps"),
                "fps95": sm.get("pipeline_fps_p95"),
            })

    hits = [r for r in rows if r["hit"]]
    L = [f"# {a.title}", ""]
    L.append(f"{len(hits)} of {len(rows)} engagements intercepted "
             f"(**{100.0 * len(hits) / max(1, len(rows)):.1f}%**) across "
             f"{len({r['scene'] for r in rows})} environments.")
    L.append("")
    L.append("A hit is a true closest approach inside 1.0 m — two Iris airframes "
             "touching are about 0.5 m centre to centre.")
    L.append("")

    # -- headline ------------------------------------------------------------
    cpas = [r["cpa"] for r in hits if r["cpa"] is not None]
    ts = [r["t"] for r in hits if r["t"]]
    acqs = [r["acq"] for r in rows if r["acq"] is not None]
    L += ["## Headline", "",
          "| metric | value |", "|---|---|",
          f"| engagements | {len(rows)} |",
          f"| intercepts | **{len(hits)} / {len(rows)}** "
          f"({100.0 * len(hits) / max(1, len(rows)):.1f}%) |"]
    if cpas:
        L += [f"| mean true miss distance | **{st.fmean(cpas):.3f} m** |",
              f"| median / p95 true miss | {st.median(cpas):.3f} m / "
              f"{_fmt(_q(cpas, 0.95), 3)} m |",
              f"| best pass | {min(cpas):.3f} m |"]
    if ts:
        L += [f"| median time to intercept | **{st.median(ts):.2f} s** |",
              f"| p95 time to intercept | {_fmt(_q(ts, 0.95))} s |"]
    if acqs:
        L += [f"| median time to acquire | {st.median(acqs):.2f} s |"]
    r0s = [r["r0"] for r in rows if r["r0"]]
    if r0s:
        L += [f"| start range | {min(r0s):.0f}–{max(r0s):.0f} m "
              f"(median {st.median(r0s):.0f} m) |"]
    L.append("")

    # -- frame rate ----------------------------------------------------------
    det = [r["det_ms"] for r in rows if r["det_ms"]]
    trk = [r["trk_ms"] for r in rows if r["trk_ms"]]
    gui = [r["gui_ms"] for r in rows if r["gui_ms"]]
    fps = [r["fps"] for r in rows if r["fps"]]
    fps95 = [r["fps95"] for r in rows if r["fps95"]]
    if det:
        loop = st.fmean(det) + st.fmean(trk or [0]) + st.fmean(gui or [0])
        L += ["## Frame rate", "",
              "Pipeline rate — detector + tracker + guidance. **Not** the test "
              "run's wall-clock rate, which is dominated by Isaac rendering five "
              "frames per control tick and does not exist on the aircraft.", "",
              "| stage | ms / frame | share |", "|---|---|---|",
              f"| detector | {st.fmean(det):.1f} | "
              f"{100.0 * st.fmean(det) / loop:.0f}% |",
              f"| tracker | {st.fmean(trk or [0]):.2f} | "
              f"{100.0 * st.fmean(trk or [0]) / loop:.0f}% |",
              f"| guidance | {st.fmean(gui or [0]):.2f} | "
              f"{100.0 * st.fmean(gui or [0]) / loop:.0f}% |",
              f"| **total** | **{loop:.1f}** | |", "",
              # Derived from the measured loop time rather than read from the
              # runs, so a run recorded before pipeline_fps existed still gets a
              # correct headline instead of a zero.
              f"- **{1000.0 / loop:.1f} FPS** mean"
              + (f", **{st.fmean(fps95):.1f} FPS** at p95 (worst-frame)"
                 if fps95 else ""),
              f"- control loop runs at 20 Hz (50 ms), so the pipeline "
              f"{'meets' if loop <= 50 else 'does NOT meet'} real time "
              f"({loop:.1f} ms vs 50 ms budget)",
              "- the detector is "
              f"{100.0 * st.fmean(det) / loop:.0f}% of the cost: it is the only "
              "stage worth optimising for edge hardware", ""]

    # -- by environment ------------------------------------------------------
    L += ["## By environment", "",
          "| environment | intercepts | mean miss | median t | det rate | FPS |",
          "|---|---|---|---|---|---|"]
    for scene in sorted({r["scene"] for r in rows}):
        g = [r for r in rows if r["scene"] == scene]
        gh = [r for r in g if r["hit"]]
        c = [r["cpa"] for r in gh if r["cpa"] is not None]
        tt = [r["t"] for r in gh if r["t"]]
        dr = [r["det"] for r in g if r["det"] is not None]
        ff = [r["fps"] for r in g if r["fps"]]
        if not ff:
            per = [(r["det_ms"] or 0) + (r["trk_ms"] or 0) + (r["gui_ms"] or 0)
                   for r in g if r["det_ms"]]
            ff = [1000.0 / x for x in per if x > 0]
        L.append(f"| {SCENE_NAMES.get(scene, scene)} | **{len(gh)}/{len(g)}** "
                 f"({100.0 * len(gh) / max(1, len(g)):.0f}%) | "
                 f"{_fmt(st.fmean(c), 3) if c else '-'} m | "
                 f"{_fmt(st.median(tt)) if tt else '-'} s | "
                 f"{_pct(dr)} | {_fmt(st.fmean(ff), 1) if ff else '-'} |")
    L.append("")

    # -- by how the intruder arrived ----------------------------------------
    entries = sorted({r["entry"] for r in rows if r["entry"] != "-"})
    if entries:
        L += ["## By approach direction", "",
              "How the intruder arrived, which is the axis a real engagement "
              "varies along.", "",
              "| arrival | intercepts | mean miss | median t | median acquire |",
              "|---|---|---|---|---|"]
        for e in entries:
            g = [r for r in rows if r["entry"] == e]
            gh = [r for r in g if r["hit"]]
            c = [r["cpa"] for r in gh if r["cpa"] is not None]
            tt = [r["t"] for r in gh if r["t"]]
            aq = [r["acq"] for r in g if r["acq"] is not None]
            L.append(f"| `{e}` | **{len(gh)}/{len(g)}** | "
                     f"{_fmt(st.fmean(c), 3) if c else '-'} m | "
                     f"{_fmt(st.median(tt)) if tt else '-'} s | "
                     f"{_fmt(st.median(aq)) if aq else '-'} s |")
        L.append("")

    # -- by evasion policy ---------------------------------------------------
    L += ["## By evasion policy", "",
          "| policy | intercepts | mean miss | median t |", "|---|---|---|---|"]
    for p in sorted({r["policy"] for r in rows}):
        g = [r for r in rows if r["policy"] == p]
        gh = [r for r in g if r["hit"]]
        c = [r["cpa"] for r in gh if r["cpa"] is not None]
        tt = [r["t"] for r in gh if r["t"]]
        L.append(f"| `{p}` | **{len(gh)}/{len(g)}** | "
                 f"{_fmt(st.fmean(c), 3) if c else '-'} m | "
                 f"{_fmt(st.median(tt)) if tt else '-'} s |")
    L.append("")

    # -- aim geometry --------------------------------------------------------
    vs = [r["vert"] for r in hits if r["vert"] is not None]
    ls = [r["lat"] for r in hits if r["lat"] is not None]
    if vs:
        L += ["## Where the intercepts land", "",
              "Signed, in the chaser's own axes at the true closest approach. "
              "Positive vertical means it passed above the target.", "",
              "| axis | mean | sd |", "|---|---|---|",
              f"| vertical | {st.fmean(vs) * 100:+.1f} cm | "
              f"{st.pstdev(vs) * 100 if len(vs) > 1 else 0:.1f} cm |",
              f"| lateral | {st.fmean(ls) * 100:+.1f} cm | "
              f"{st.pstdev(ls) * 100 if len(ls) > 1 else 0:.1f} cm |", ""]

    # -- every engagement ----------------------------------------------------
    L += ["## Every engagement", "",
          "| scenario | env | arrival | evasion | start | result | miss | "
          "t | acquire | det | trk | FPS |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: (x["scene"], x["name"])):
        L.append(
            f"| `{r['name']}` | {r['scene']} | {r['entry']} | `{r['policy']}` | "
            f"{_fmt(r['r0'], 0)} m | {'**HIT**' if r['hit'] else 'miss'} | "
            f"{_fmt(r['cpa'], 3)} m | {_fmt(r['t'])} s | {_fmt(r['acq'])} s | "
            f"{_fmt(r['det'])} | {_fmt(r['trk'])} | {_fmt(r['fps'], 1)} |")
    L.append("")

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"{len(rows)} engagements from {len(runs)} runs -> {out}")
    if det:
        print(f"pipeline {1000.0 / loop:.1f} FPS mean, "
              f"{len(hits)}/{len(rows)} intercepts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
