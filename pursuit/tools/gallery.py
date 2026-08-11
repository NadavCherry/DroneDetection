"""Collect every pursuit video into one browsable place.

Runs scatter their output: a suite per directory, a directory per scene and
detector, and after a few rounds of experiments the interesting clips are spread
across a dozen folders with no way to see them side by side. This gathers them
under a single root -- hardlinked, so it costs no disk and no copy time -- and
writes an ``index.html`` that plays them inline with the outcome of each run
attached.

Each clip is labelled with what actually happened, read from the run's own
``summary.json`` or telemetry rather than from the filename, because the
filename says which scenario was *requested* and the summary says whether it hit.

    python -m pursuit.tools.gallery --out work/pursuit/gallery
    python -m pursuit.tools.gallery --out work/pursuit/gallery --open
"""
from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCENES = {"town": "rivermark", "rivermark": "rivermark", "sky": "skydome",
          "skydome": "skydome"}


def _guess(path: Path, keys: dict, default: str) -> str:
    low = str(path).lower()
    for key, value in keys.items():
        if key in low:
            return value
    return default


def _outcomes(run_dir: Path) -> dict:
    """scenario name -> {outcome, miss, det, ...} from whatever the run left."""
    out: dict = {}
    for cand in ("summary.json", "results.json", "report.json"):
        p = run_dir / cand
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        rows = data.get("results") if isinstance(data, dict) else data
        if isinstance(rows, list):
            for r in rows:
                if isinstance(r, dict) and r.get("name"):
                    out[str(r["name"])] = r
    return out


def _from_telemetry(tel: Path) -> Optional[dict]:
    """Recover an outcome from the per-frame log when no summary was written."""
    try:
        rows = json.loads(tel.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(rows, list) or not rows:
        return None
    ranges = [r.get("range_true") for r in rows
              if isinstance(r, dict) and r.get("range_true") is not None]
    if not ranges:
        return None
    scored = [r for r in rows if isinstance(r, dict) and r.get("score")]
    return {"miss_distance_m": min(ranges),
            "frames": len(rows),
            "detect_rate": round(len(scored) / max(1, len(rows)), 3),
            "outcome": "intercept" if min(ranges) <= 1.5 else "timeout",
            "recovered": True}


CSS = """
:root{--bg:#0f1115;--fg:#e6e8eb;--dim:#9aa3ad;--card:#171a20;--line:#242833;
--hit:#3ddc84;--miss:#ff6b6b}
@media (prefers-color-scheme:light){:root{--bg:#f7f8fa;--fg:#14171c;--dim:#5b6570;
--card:#fff;--line:#e2e5ea}}
*{box-sizing:border-box}
body{margin:0;padding:28px;background:var(--bg);color:var(--fg);
font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
h1{font-size:22px;margin:0 0 4px}
h2{font-size:16px;margin:32px 0 12px;padding-bottom:8px;border-bottom:1px solid var(--line)}
.sub{color:var(--dim);margin-bottom:24px}
.grid{display:grid;gap:16px;grid-template-columns:repeat(auto-fill,minmax(340px,1fr))}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
overflow:hidden}
video{width:100%;display:block;background:#000;aspect-ratio:12/7}
.meta{padding:10px 12px}
.name{font-weight:600;font-size:14px;display:flex;justify-content:space-between;
align-items:center;gap:8px}
.tag{font-size:11px;font-weight:700;padding:2px 7px;border-radius:99px;
letter-spacing:.03em}
.hit{background:var(--hit);color:#06301a}
.miss{background:var(--miss);color:#3d0b0b}
.facts{color:var(--dim);font-size:12.5px;margin-top:4px;font-variant-numeric:tabular-nums}
.filters{margin:0 0 20px;display:flex;flex-wrap:wrap;gap:8px}
button{background:var(--card);color:var(--fg);border:1px solid var(--line);
border-radius:99px;padding:5px 13px;font-size:13px;cursor:pointer}
button.on{background:var(--fg);color:var(--bg);border-color:var(--fg)}
"""

JS = """
const btns=[...document.querySelectorAll('button[data-f]')];
btns.forEach(b=>b.onclick=()=>{
  btns.forEach(x=>x.classList.toggle('on',x===b));
  const f=b.dataset.f;
  document.querySelectorAll('.card').forEach(c=>{
    c.style.display=(f==='all'||c.dataset.tags.includes(f))?'':'none';});
  document.querySelectorAll('h2,.grid').forEach(s=>{
    if(s.tagName==='H2')return;
    const any=[...s.querySelectorAll('.card')].some(c=>c.style.display!=='none');
    s.style.display=any?'':'none';
    if(s.previousElementSibling&&s.previousElementSibling.tagName==='H2')
      s.previousElementSibling.style.display=any?'':'none';});
});
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--search", default="work", help="root to scan for *.mp4")
    ap.add_argument("--out", default="work/pursuit/gallery")
    ap.add_argument("--copy", action="store_true",
                    help="copy instead of hardlink (use across filesystems)")
    ap.add_argument("--min-kb", type=float, default=80.0,
                    help="skip clips smaller than this (failed encodes)")
    a = ap.parse_args(argv)

    search, out = ROOT / a.search, ROOT / a.out
    videos = sorted(p for p in search.rglob("*.mp4") if out not in p.parents)
    if not videos:
        print(f"no videos under {search}")
        return 1

    out.mkdir(parents=True, exist_ok=True)
    (out / "clips").mkdir(exist_ok=True)

    groups: dict[str, list] = {}
    summaries: dict[Path, dict] = {}
    skipped = 0
    for src in videos:
        if src.stat().st_size < a.min_kb * 1024:
            skipped += 1
            continue
        run = src.parent
        if run not in summaries:
            summaries[run] = _outcomes(run)
        info = dict(summaries[run].get(src.stem) or {})
        if not info:
            tel = run / f"{src.stem}.telemetry.json"
            if tel.exists():
                info = _from_telemetry(tel) or {}

        scene = _guess(src, SCENES, "skydome")
        detector = _guess(src, {"oracle": "oracle", "fusion": "fusion",
                                "yolo": "yolo"}, "yolo")
        label = f"{run.name}__{src.stem}.mp4"
        dst = out / "clips" / label
        if not dst.exists():
            if a.copy:
                shutil.copy2(src, dst)
            else:
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copy2(src, dst)
        groups.setdefault(run.name, []).append((src.stem, dst, info, scene,
                                                detector))

    total = sum(len(v) for v in groups.values())
    parts = [f"<title>Pursuit runs</title><style>{CSS}</style>",
             "<h1>Pursuit runs</h1>",
             f"<div class=sub>{total} clips from {len(groups)} runs. "
             "Stage 1 detect and track, stage 2 close to collision.</div>",
             "<div class=filters><button class=on data-f=all>all</button>"
             "<button data-f=rivermark>rivermark</button>"
             "<button data-f=skydome>skydome</button>"
             "<button data-f=hit>hits</button>"
             "<button data-f=miss>misses</button></div>"]

    for run in sorted(groups, key=lambda r: (-len(groups[r]), r)):
        clips = sorted(groups[run])
        hits = sum(1 for _n, _d, i, _s, _x in clips
                   if i.get("outcome") == "intercept")
        parts.append(f"<h2>{html.escape(run)} &middot; {hits}/{len(clips)} "
                     f"intercepts</h2><div class=grid>")
        for name, dst, info, scene, detector in clips:
            hit = info.get("outcome") == "intercept"
            tag = ("<span class='tag hit'>HIT</span>" if hit
                   else "<span class='tag miss'>MISS</span>" if info
                   else "")
            facts = []
            if info.get("miss_distance_m") is not None:
                facts.append(f"miss {info['miss_distance_m']:.2f} m")
            if info.get("time_to_intercept_s"):
                facts.append(f"t {info['time_to_intercept_s']:.1f} s")
            if info.get("detect_rate") is not None:
                facts.append(f"det {info['detect_rate']:.2f}")
            if info.get("reveal_time_s") is not None:
                facts.append(f"reveal {info['reveal_time_s']:.1f} s")
            facts.append(scene)
            tags = f"{scene} {detector} {'hit' if hit else 'miss'}"
            parts.append(
                f"<div class=card data-tags='{tags}'>"
                f"<video src='clips/{html.escape(dst.name)}' controls preload=none "
                f"playsinline></video><div class=meta><div class=name>"
                f"<span>{html.escape(name)}</span>{tag}</div>"
                f"<div class=facts>{html.escape(' &middot; '.join(facts))}</div>"
                f"</div></div>")
        parts.append("</div>")
    parts.append(f"<script>{JS}</script>")

    index = out / "index.html"
    index.write_text("\n".join(parts).replace("&amp;middot;", "&middot;"))
    print(f"{total} clips from {len(groups)} runs -> {index}")
    if skipped:
        print(f"skipped {skipped} clip(s) under {a.min_kb:.0f} KB")
    print(f"open: file://{index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
