#!/usr/bin/env python3
"""Check every number in the launch package against the artefact that produced it.

WHY THIS EXISTS
---------------
The LinkedIn post is the highest-stakes artefact in this repository. It is the only one
most readers will ever see, it cannot be quietly corrected after the fact, and it is the
one place where a stale number does reputational rather than merely technical damage.

This project has already shipped a social card claiming "AP/F1 1.000 on unseen real video;
24/24 intercepted" -- both retracted by its own README, both live on the public site for
weeks, because nothing checked the public-facing text against the evidence. That is exactly
the failure mode a launch is most exposed to: the numbers move, the prose does not.

So the post gets the same treatment as a results table. Every claim in
`docs/launch/linkedin.md` that can be checked is checked here, against the JSON the
evaluation jobs wrote, and this refuses to pass if any of them has drifted.

    PYTHONPATH=. python tools/check_launch_claims.py

Exit code is 0 only if every claim matches. Run it before posting, and again after any
re-evaluation.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


# The sentences section 6 of the launch package forbids, as patterns a regex can honestly
# judge. Module level so `dronedet/tests/test_check_launch_claims.py` can assert that each
# one still matches the sentence it exists to catch.
#
# HOW THIS LIST FAILED ITS OWN PURPOSE ONCE: it was first written through a NON-raw Python
# string, so every \b word-boundary was interpreted at write time and landed in the file as
# byte 0x08 -- invisible in cat, in an editor, and in review. The patterns searched for a
# backspace character, matched nothing, and all five checks reported PASS against a post
# that contained all five phrases. A check that cannot fail is worse than no check, because
# it gets reported as evidence. The test asserts both that the bytes are clean and that the
# patterns still fire on dishonest text.
FORBIDDEN = [
    (r'\bstate of the art\b', 'claims SOTA'),
    (r'\bzero false positives\b', 'score-weighted per-frame metric only'),
    (r'\bbeats YOLOMG\b', 'not supported: it leads overall'),
    (r'\bdt\s*=\s*6 is optimal\b', 'the sweep does not establish this'),
    (r'\b24\s*/\s*24\b(?![^.]{0,140}(?:perfect sensor|oracle|guidance))',
     '24/24 without naming the perfect sensor'),
]


def publishable(post: str) -> str:
    """The blockquoted text -- exactly what gets pasted into LinkedIn.

    Scoping the scan to this is not a convenience, it is correctness. A whole-document scan
    is wrong on its face: SS1 offers the headline "My detector loses to the state of the
    art", SS5 asks "Did you beat the state of the art?", and SS6 IS the list of forbidden
    phrases with every one of them in its 'do not write' column. All three are legitimate
    and all three trip a naive scan. Editorial commentary about the post is not the post.
    """
    return "\n".join(ln.lstrip()[1:].strip()
                     for ln in post.splitlines() if ln.lstrip().startswith(">"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--post", type=Path, default=REPO / "docs/launch/linkedin.md")
    ap.add_argument("--results", type=Path, default=REPO / "work")
    a = ap.parse_args()

    if not a.post.is_file():
        print(f"no launch package at {a.post}")
        return 1
    post = a.post.read_text(encoding="utf-8")

    passed: list[str] = []
    failed: list[str] = []

    def check(label: str, cond: bool, detail: str = "") -> None:
        (passed if cond else failed).append(f"{label:44s} {detail}")

    # ---------------------------------------------------------------- the ablation
    rep = a.results / "ablation" / "REPORT.md"
    if rep.is_file():
        t = rep.read_text(encoding="utf-8")
        check("controlled ablation 0.159 -> 0.895",
              "0.159" in t and "0.895" in t, str(rep.relative_to(REPO)))
    else:
        check("controlled ablation 0.159 -> 0.895", False, "REPORT.md missing")

    # ---------------------------------------------------------------- edge speed
    eb = load(a.results / "reports" / "edge" / "edge_bench.json")
    if eb:
        def arm(backend, sz):
            return next((r for r in eb
                         if r.get("backend") == backend and r.get("imgsz") == sz), None)
        e, p = arm("engine", 1280), arm("pt", 1280)
        if e:
            check("58.9 fps, engine @1280",
                  abs(e["steady_state"]["fps_at_p50"] - 58.9) < 0.05,
                  "actual %.1f" % e["steady_state"]["fps_at_p50"])
            check("AP 0.876, engine @1280", abs(e["ap"] - 0.876) < 0.001,
                  "actual %.4f" % e["ap"])
        if p:
            check("35.2 fps, .pt fallback @1280",
                  abs(p["steady_state"]["fps_at_p50"] - 35.2) < 0.05,
                  "actual %.1f" % p["steady_state"]["fps_at_p50"])
        # The post must NOT resurrect the unreproduced figure.
        check("no 100+ fps claim in the post",
              not re.search(r"\b1\d\d\s*(?:\+\s*)?fps\b", post, re.I)
              or "did **not** reproduce" in post or "did NOT reproduce" in post,
              "an old 104 fps number must never come back unqualified")
    else:
        check("edge bench artefact present", False, "work/reports/edge/edge_bench.json")

    # ---------------------------------------------------------------- birds
    tb = load(a.results / "reports" / "tracks" / "pcmax_0705.json")
    if tb:
        check("934 labelled bird instances",
              tb["ground_truth"]["bird_instances"] == 934,
              str(tb["ground_truth"]["bird_instances"]))
        check("440 detections land on birds",
              tb["raw_detections"]["on_bird"] == 440,
              str(tb["raw_detections"]["on_bird"]))
        check("0 birds raised as targets",
              tb["operating_point"]["bird_false_alarms"] == 0,
              str(tb["operating_point"]["bird_false_alarms"]))
        # The counterpart must be present too: quoting the bird result without the
        # clutter result is the exact selective-reporting this package forbids.
        n_clutter = tb["operating_point"]["clutter_false_alarms"]
        check("clutter counterpart stated in the post",
              str(n_clutter) in post,
              f"{n_clutter} clutter tracks must appear beside the bird claim")
    else:
        check("track-level artefact present", False, "work/reports/tracks/pcmax_0705.json")

    # ---------------------------------------------------------------- the size curve
    sc = load(a.results / "reports" / "size_curve" / "ardmav_mission.json")
    if sc:
        arms = sc["arms"]
        for b in ("<8 px", "8-10 px"):
            if b in arms.get("ours", {}).get("bins", {}):
                d = arms["ours"]["bins"][b]["mean"] - arms["yolomg"]["bins"][b]["mean"]
                check(f"we lead {b} in the mean", d > 0, "delta %+.3f" % d)
        sig_small = [r for r in sc.get("paired", [])
                     if r["bin"] in ("<8 px", "8-10 px") and r["significant"]]
        # The load-bearing honesty claim: if this EVER becomes significant, the post's
        # wording must change, and this check is what will catch it.
        check("sub-10 px lead is NOT significant", len(sig_small) == 0,
              f"{len(sig_small)} significant -- the post says it is a trend, not a result")
        check("post calls it a trend, not a result",
              "trend" in post.lower() and "not a result" in post.lower())
    else:
        check("size curve artefact present", False, "ardmav_mission.json")

    # ---------------------------------------------------------------- the figures
    figdir = REPO / "docs" / "media" / "paper"
    for name in sorted(set(re.findall(r"`(fig\d_[a-z0-9_]+\.png)`", post))):
        check(f"carousel figure {name}", (figdir / name).is_file())
    # Slide 1 is not a fig*.png -- it is a repo asset by full path. It was unchecked until
    # a run showed only four of the five carousel slides being verified.
    for rel in sorted(set(re.findall(r"`(docs/media/[a-z0-9_/]+\.(?:jpg|png|gif))`", post))):
        check(f"carousel asset {rel}", (REPO / rel).is_file())

    # ---------------------------------------------------------------- forbidden phrasing
    # Scope and rationale: see FORBIDDEN and publishable() at module level.
    published = publishable(post)
    check("publishable text located to scan", len(published) > 1500,
          "%d chars of blockquote -- if this collapses, every check below is vacuous"
          % len(published))
    for pat, why in FORBIDDEN:
        hits = [h for h in re.findall(pat, published, re.I) if str(h).strip()]
        check('forbidden phrasing absent: ' + why, not hits,
              str(hits[:2]) if hits else '')
    print("PASSED (%d)" % len(passed))
    for s in passed:
        print("   ", s)
    if failed:
        print("\nFAILED (%d)" % len(failed))
        for s in failed:
            print("   ", s)
        print("\nThe launch package disagrees with the evidence. Fix the post, or the "
              "artefact, before anything is published.")
        return 1
    print("\nEvery checkable claim in the launch package matches its artefact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
