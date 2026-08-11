#!/usr/bin/env python3
"""Build ``docs/gallery.html`` from ``docs/media/showcase.json``.

The manifest is written by ``tools/publish_showcase.py``, which reads each
clip's facts out of the run that produced it. Generating the page from that
manifest rather than writing it by hand is the whole point: a caption on the
site cannot drift away from the engagement it describes, and re-running the
publisher after a new campaign updates the page for free.

    .venv/bin/python tools/publish_showcase.py && .venv/bin/python tools/make_gallery.py

The page shares ``docs/index.html``'s design tokens; keep the two palettes in
step if either is edited.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

# showcase.json stores paths relative to docs/media/; the page lives one level up.
MEDIA = "media/"

# Full-length runs that live in docs/media already -- not produced by the
# publisher, so they are described here.
DETECTION = [
    dict(mp4="media/10_06_baseline_vs_pcmax_vs_edgert.mp4",
         title="Baseline &#124; PC-MAX &#124; EDGE-RT, side by side",
         blurb="The same 19.7 s of unseen test video through three systems at once. Left: a "
               "YOLO26n trained on a real multi-scene drone dataset. Middle and right: the two "
               "shipped models.",
         facts="1280&#215;720 &#183; 591 frames &#183; the drone is 3&#8211;14 px throughout"),
    dict(mp4="media/10_06_pcmax_tracks.mp4",
         title="PC-MAX on the unseen test video",
         blurb="The accuracy-first desktop profile: three detection streams fused by centre "
               "agreement, then a Kalman tracker and a track-level classifier.",
         facts="AP / F1 = 1.000 &#183; zero false positives &#183; ~4 fps on an RTX 5070"),
    dict(mp4="media/10_06_edgert_tracks.mp4",
         title="EDGE-RT on the unseen test video",
         blurb="The whole pipeline distilled into one YOLOv8-nano-P2 reading the stabilised "
               "3-moment stack full-frame, under TensorRT FP16. One network is the pipeline.",
         facts="AP / F1 = 1.000 &#183; ~74 fps end to end"),
    dict(mp4="media/10_06_baseline_dets.mp4",
         title="The baseline on the same video",
         blurb="Single-frame appearance. Excellent against open sky in the final second "
               "(conf 0.6&#8211;0.84, zero false positives) and silent everywhere else, even at "
               "conf 0.02.",
         facts="12.5% flight coverage &#183; nothing below the treeline"),
    dict(mp4="media/07_05_round2_tracks.mp4",
         title="The training video, with the hand labels painted",
         blurb="Every frame of 07_05 was labelled by hand; this is the authoritative ground "
               "truth the detection numbers are scored against.",
         facts="the labeller's method is the algorithm's: flip frames, watch what moves"),
]

HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Video gallery &mdash; SpeckLock</title>
<meta name="description" content="Every recorded engagement: the four-camera city defence ring, the one-camera pursuit campaign, and the full-length detection runs on real video.">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>&#127981;</text></svg>">
<style>
:root{
  color-scheme: light;
  --plane:#f9f9f7; --surface:#fcfcfb; --surface-2:#f2f1ed;
  --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781; --hair:rgba(11,11,11,.10);
  --s1:#2a78d6; --good:#0ca30c; --crit:#d03b3b; --sim:#eda100; --real:#0ca30c;
  --shadow:0 1px 2px rgba(11,11,11,.05), 0 8px 24px rgba(11,11,11,.05);
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  color-scheme: dark;
  --plane:#0d0d0d; --surface:#1a1a19; --surface-2:#232322;
  --ink:#fff; --ink-2:#c3c2b7; --muted:#898781; --hair:rgba(255,255,255,.10);
  --s1:#3987e5; --sim:#c98500;
  --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px rgba(0,0,0,.35);
}}
:root[data-theme="dark"]{
  color-scheme: dark;
  --plane:#0d0d0d; --surface:#1a1a19; --surface-2:#232322;
  --ink:#fff; --ink-2:#c3c2b7; --muted:#898781; --hair:rgba(255,255,255,.10);
  --s1:#3987e5; --sim:#c98500;
}
*{box-sizing:border-box}
body{margin:0;background:var(--plane);color:var(--ink);
  font:16px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;-webkit-font-smoothing:antialiased}
.wrap{max-width:1240px;margin:0 auto;padding:0 22px}
a{color:var(--s1)}
nav{position:sticky;top:0;z-index:40;backdrop-filter:blur(10px);
  background:color-mix(in srgb, var(--plane) 88%, transparent);border-bottom:1px solid var(--hair)}
nav .wrap{display:flex;align-items:center;gap:20px;height:56px}
nav a.brand{font-weight:700;color:var(--ink);text-decoration:none}
nav .sp{flex:1}
nav a.l{color:var(--ink-2);text-decoration:none;font-size:14.5px}
nav a.l:hover{color:var(--ink)}
header{padding:48px 0 10px}
h1{font-size:clamp(30px,4.4vw,44px);letter-spacing:-.02em;margin:0 0 12px}
.lede{color:var(--ink-2);max-width:76ch;font-size:17.5px;margin:0 0 8px}
h2{font-size:24px;letter-spacing:-.015em;margin:52px 0 4px}
h2 + p{color:var(--ink-2);max-width:80ch;margin:0 0 22px}
.chip{display:inline-flex;align-items:center;gap:8px;font-size:13px;font-weight:600;
  border:1px solid var(--hair);border-radius:999px;padding:5px 13px 5px 10px;background:var(--surface);
  vertical-align:middle;margin-left:10px}
.chip .dot{width:9px;height:9px;border-radius:50%}
.grid{display:grid;gap:20px;grid-template-columns:repeat(auto-fill,minmax(400px,1fr))}
.grid.wide{grid-template-columns:repeat(auto-fill,minmax(520px,1fr))}
.card{background:var(--surface);border:1px solid var(--hair);border-radius:14px;
  overflow:hidden;box-shadow:var(--shadow)}
video{width:100%;display:block;background:#000}
.m{padding:14px 17px 16px}
.m .t{font-weight:650;font-size:15.5px;display:flex;justify-content:space-between;
  align-items:baseline;gap:10px}
.m .b{font-size:14px;color:var(--ink-2);margin-top:6px}
.m .f{font-size:12.5px;color:var(--muted);margin-top:9px;font-variant-numeric:tabular-nums;
  border-top:1px solid var(--hair);padding-top:9px}
.tag{font-size:11px;font-weight:800;letter-spacing:.05em;padding:3px 9px;border-radius:999px;flex:none}
.tag.hit{background:var(--good);color:#052e05}
.tag.miss{background:var(--crit);color:#2e0505}
footer{border-top:1px solid var(--hair);margin-top:56px;padding:34px 0 60px;
  color:var(--ink-2);font-size:14.5px}
footer a{text-decoration:none}
</style>
</head>
<body>
<nav><div class="wrap">
  <a class="brand" href="index.html">SpeckLock</a>
  <a class="l" href="index.html#method">Method</a>
  <a class="l" href="index.html#results">Results</a>
  <span class="sp"></span>
  <a class="l" href="https://github.com/NadavCherry/SpeckLock">GitHub &#8599;</a>
</div></nav>
<div class="wrap">
<header>
  <h1>Video gallery</h1>
  <p class="lede">Every recorded engagement, with the facts taken from the run that produced
  it. Interception clips are Isaac Sim, closed loop at 20&nbsp;Hz; detection clips are real
  video. Nothing here is a re-cut of a good moment &mdash; each clip is one engagement,
  start to finish.</p>
</header>
"""

FOOT = """
<footer>
  <div><b>Nadav Cherry</b> &mdash; <a href="index.html">back to the project page</a> &middot;
  <a href="https://github.com/NadavCherry/SpeckLock">GitHub</a></div>
</footer>
</div>
</body>
</html>
"""


def card(mp4: str, poster: str | None, title: str, blurb: str, facts: str,
         tag: str | None = None) -> str:
    t = f'<span class="tag {tag.lower()}">{tag}</span>' if tag else ""
    p = f' poster="{poster}"' if poster else ""
    return (f'<div class="card">'
            f'<video src="{mp4}"{p} controls preload="none" playsinline></video>'
            f'<div class="m"><div class="t"><span>{title}</span>{t}</div>'
            f'<div class="b">{blurb}</div><div class="f">{facts}</div></div></div>')


def facts_line(f: dict) -> str:
    """One line of measured facts, only from fields the run actually reported."""
    bits = []
    if f.get("cpa_m") is not None:
        bits.append(f"closest approach <b>{f['cpa_m']:.3f} m</b>")
    if f.get("t_intercept_s") is not None:
        bits.append(f"intercepted at {f['t_intercept_s']:.2f} s")
    if f.get("acquire_s") is not None:
        bits.append(f"acquired in {f['acquire_s']:.2f} s")
    if f.get("margin_s") is not None:
        bits.append(f"{f['margin_s']:.2f} s before the strike")
    if f.get("detect_rate") is not None:
        bits.append(f"detection rate {f['detect_rate']:.2f}")
    if not bits:
        bits.append("see the run manifest in <code>work/pursuit/</code>")
    return " &#183; ".join(bits)


def main() -> int:
    manifest = json.loads((DOCS / "media" / "showcase.json").read_text())
    out = [HEAD]

    out.append('<h2>City defence &mdash; the four-camera ring'
               '<span class="chip"><span class="dot" style="background:var(--sim)"></span>'
               'Isaac Sim</span></h2>')
    out.append("<p>The interceptor rises over the town and holds station, watching all four "
               "quarters at once. An intruder arrives from a bearing drawn from the whole "
               "circle, commits to the nearest surveyed building, and does not break off when "
               "it is seen. Ten of the twenty-four scored engagements were recorded. Each frame "
               "shows all four camera feeds with the owning camera outlined, a "
               "contrast-stretched magnified inset of the target, and a top-down map with both "
               "flight paths and a red cross on the wall the intruder was aiming at.</p>")
    out.append('<div class="grid">')
    for c in manifest.get("city", []):
        f = c.get("facts", {})
        tag = "HIT" if f.get("outcome") == "intercept" else None
        out.append(card(MEDIA + c["mp4"], MEDIA + c["poster"] if c.get("poster") else None, c["title"], c["blurb"],
                        facts_line(f) + f" &#183; played {c['speed']:g}&#215;", tag))
    out.append("</div>")

    out.append('<h2>One camera, chase and intercept'
               '<span class="chip"><span class="dot" style="background:var(--sim)"></span>'
               'Isaac Sim</span></h2>')
    out.append("<p>The earlier campaign: one forward-facing camera, an intruder crossing or "
               "fleeing, and nothing to defend. 54 of 62 engagements intercepted (87.1%). "
               "These six are chosen to show the range of the campaign &mdash; including the "
               "engagement that motivated building the ring, and one of the eight failures in "
               "full.</p>")
    out.append('<div class="grid">')
    for c in manifest.get("chase", []):
        f = c.get("facts", {})
        tag = "HIT" if f.get("outcome") == "intercept" else "MISS"
        out.append(card(MEDIA + c["mp4"], MEDIA + c["poster"] if c.get("poster") else None, c["title"], c["blurb"],
                        facts_line(f) + f" &#183; played {c['speed']:g}&#215;", tag))
    out.append("</div>")

    out.append('<h2>Detection on real video'
               '<span class="chip"><span class="dot" style="background:var(--real)"></span>'
               'real video</span></h2>')
    out.append("<p>Full-length runs on the hand-labelled source footage. The source files hide "
               "their opening seconds behind an MP4 edit list that every decoder honours "
               "silently &mdash; <code>tools/recover_full_video.py</code> remuxes them back "
               "losslessly, which is why the test video is 591 frames and not 361.</p>")
    out.append('<div class="grid wide">')
    for d in DETECTION:
        out.append(card(d["mp4"], None, d["title"], d["blurb"], d["facts"]))
    out.append("</div>")

    out.append(FOOT)
    path = DOCS / "gallery.html"
    path.write_text("\n".join(out))
    n = sum(len(manifest.get(k, [])) for k in ("city", "chase")) + len(DETECTION)
    print(f"wrote {path}  --  {n} clips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
