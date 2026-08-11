#!/usr/bin/env python3
"""The whole-method figure: one picture for both halves of the repository.

``tools/make_arch_figure.py`` and ``tools/make_arch_figures_final.py`` each draw
one *detector*. This draws the system: how a 3-pixel drone in real video becomes
a bearing, how that bearing becomes a collision, and -- the part a reader asks
about first -- which numbers were measured on real video and which in a
simulator. The two are marked with different chips on purpose.

    .venv/bin/python tools/make_arch_figure_system.py
        -> docs/media/architecture_system.svg / .png

Visual language is inherited from ``make_arch_figures_final.py`` (same palette,
same card, same three-moments motif) so the four figures read as one set.

One drawing rule, learned by breaking it: **never put a ``<tspan>`` inside a
centred ``<text>``**. The renderer anchors the tspan independently and the line
walks off its own centre. Emphasis here is a whole line in a different weight.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_arch_figures_final import (  # noqa: E402
    BLUE, CARD, CYA, EDGE, GREEN, INNER, MAG, SUB, TXT, YEL, Fig,
)

PURPLE = "#a371f7"
AMBER = "#d29922"

W, H = 2040, 1322


# ---------------------------------------------------------------------------
# small drawing helpers on top of Fig
# ---------------------------------------------------------------------------

def text(F: Fig, x, y, s, size=13.5, fill=TXT, weight=400, anchor="middle", op=1.0,
         mono=False):
    fam = ' font-family="DejaVu Sans Mono, monospace"' if mono else ""
    F.svg.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" '
                 f'font-weight="{weight}" fill="{fill}" opacity="{op}"{fam}>{s}</text>')


def block(F: Fig, cx, y, lines, size=13.5, lh=20, fill=TXT, op=0.88):
    """A centred stack of plain lines. `lines` may hold (text, weight, colour)."""
    for i, ln in enumerate(lines):
        if isinstance(ln, tuple):
            s, wt, col = (ln + (fill,))[:3] if len(ln) == 2 else ln
        else:
            s, wt, col = ln, 400, fill
        text(F, cx, y + i * lh, s, size=size, fill=col, weight=wt,
             op=1.0 if wt >= 600 else op)


def panel(F: Fig, x, y, w, h, badge, badge_col, title, lines, accent=None, aw=2,
          title_dy=56, lines_dy=82, lh=20, size=13.5):
    F.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{CARD}" '
                 f'stroke="{accent or EDGE}" stroke-width="{aw}"/>')
    F.header(x, y, w, badge, badge_col)
    if title:
        text(F, x + w / 2, y + title_dy, title, size=16.5, weight=700)
    block(F, x + w / 2, y + lines_dy, lines, size=size, lh=lh)


def footnote(F: Fig, x, y, w, s, fill=SUB):
    text(F, x + w / 2, y, s, size=12.5, fill=fill)


def lane(F: Fig, y, h, num, title, chip, chip_color):
    F.svg.append(f'<rect x="20" y="{y}" width="{W - 40}" height="{h}" rx="14" '
                 f'fill="#11161d" stroke="{EDGE}" stroke-width="1.5"/>')
    F.svg.append(f'<circle cx="58" cy="{y + 32}" r="17" fill="{EDGE}"/>')
    text(F, 58, y + 38, num, size=17, weight=700)
    text(F, 88, y + 39, title, size=19.5, weight=700, anchor="start")
    cw = int(7.6 * len(chip)) + 40
    F.svg.append(f'<rect x="{W - 40 - cw}" y="{y + 16}" width="{cw}" height="30" rx="15" '
                 f'fill="none" stroke="{chip_color}" stroke-width="1.6"/>')
    F.svg.append(f'<circle cx="{W - 40 - cw + 19}" cy="{y + 31}" r="5.5" fill="{chip_color}"/>')
    text(F, W - 40 - cw + 32, y + 36, chip, size=13, fill=chip_color, weight=600,
         anchor="start")


def flow(F: Fig, boxes, y):
    for (x1, w1), (x2, _) in zip(boxes, boxes[1:]):
        F.arrow([(x1 + w1 + 7, y), (x2 - 9, y)])


# ---------------------------------------------------------------------------

def main() -> int:
    F = Fig(W, H)

    text(F, 40, 54, "See the drone, then hit it", size=31, weight=700, anchor="start")
    text(F, 40, 82,
         "One camera system &#183; no radar, no datalink, no GPS on the target "
         "&#183; the target is 3&#8211;14 pixels across",
         size=15, fill=SUB, anchor="start")
    text(F, W - 40, 48, "the whole method, end to end", size=17, weight=700, anchor="end")
    text(F, W - 40, 72, "github.com/NadavCherry/SpeckLock", size=13, fill=SUB,
         anchor="end")

    # =======================================================================
    # 1 -- detection, on real video
    # =======================================================================
    LY, LH = 108, 302
    lane(F, LY, LH, "1",
         "SEE &#8212; a drone 4 pixels wide is invisible in any single frame",
         "measured on real video", GREEN)
    cy = LY + 66
    CH = 162
    boxes = [(44, 200), (292, 218), (558, 384), (990, 296), (1334, 296), (1678, 318)]

    panel(F, 44, cy, 200, CH, "INPUT", BLUE, "video",
          ["1280 &#215; 720, 30 fps", "drone 3&#8211;14 px", "the camera moves too"])
    panel(F, 292, cy, 218, CH, "STAGE 0", EDGE, "stabilise",
          ["phase correlation", "global camera motion", "removed, frame by frame"])

    # --- the one idea everything else follows from ---
    x, w = 558, 384
    F.svg.append(f'<rect x="{x}" y="{cy}" width="{w}" height="{CH}" rx="10" fill="{CARD}" '
                 f'stroke="{GREEN}" stroke-width="3"/>')
    F.header(x, cy, w, "STAGE 1 &#183; THE CORE IDEA", GREEN)
    F.squares_motif(x + 22, cy + 62)
    tx = x + 230
    text(F, tx, cy + 62, "three moments, one image", size=16, weight=700)
    for i, (lab, col) in enumerate([("t-12", YEL), ("t-6", MAG), ("now", CYA)]):
        text(F, x + 152 + i * 52, cy + 88, lab, size=14.5, weight=700, fill=col,
             anchor="start")
    text(F, x + 310, cy + 88, "= R, G, B", size=13.5, anchor="start", op=0.85)
    block(F, tx, cy + 116,
          ["the static world cancels to grey;", "only what moved keeps its colour"])
    footnote(F, x, cy + CH + 18, w,
             "this is also how the human labeller found it: flip frames, watch what moves")

    panel(F, 990, cy, 296, CH, "STAGE 2", EDGE, "tiny-object detector",
          ["YOLOv8 with a P2 head", "NWD loss for tiny boxes", "labels inflated to 24 px"])
    panel(F, 1334, cy, 296, CH, "STAGE 3", EDGE, "track, then decide",
          ["Kalman tracker feeding a", "track-level classifier", "announced at the 8th hit"])
    panel(F, 1678, cy, 318, CH, "RESULT", BLUE, "AP / F1 = 1.000",
          [("on the unseen test video", 600, TXT), "zero false positives",
           "4 fps desktop &#183; 74 fps edge"])
    flow(F, boxes, cy + 81)

    text(F, W / 2, LY + LH - 22,
         "Same network, same recipe: single-frame input scores mAP50 0.06 and the "
         "temporal stack scores 0.83. The representation is the breakthrough, not the network.",
         size=14, fill=SUB)

    # tie the two halves together
    F.arrow([(750, LY + LH), (750, LY + LH + 34)])
    text(F, 768, LY + LH + 26,
         "the same finding on a different mount: at 3 px, only motion sees it",
         size=13, fill=SUB, anchor="start")

    # =======================================================================
    # 2 -- the seeker
    # =======================================================================
    LY, LH = 444, 330
    lane(F, LY, LH, "2",
         "SEEK &#8212; the same idea on an interceptor: four cameras, 360&#176;, "
         "one drone among ~50 contacts",
         "simulated sensor (Isaac Sim)", AMBER)
    cy = LY + 66
    CH = 196
    boxes = [(44, 240), (324, 306), (670, 250), (960, 348), (1348, 272), (1660, 336)]

    panel(F, 44, cy, 240, CH, "SENSOR", BLUE, "four cameras",
          ["96&#176; each, 90&#176; apart", "384&#176; of 360 &#8212; every seam",
           "overlaps by 6&#176; on purpose", "16.1 px/deg, same as one"])
    panel(F, 324, cy, 306, CH, "MOTION &#183; PROPOSES", EDGE, "background model",
          ["it holds station, so the four", "cameras are stationary &#8212; and a",
           "per-pixel model sees the whole", "contrast, not the moving sliver",
           ("finds 3 px at 140 m", 600, TXT)])
    panel(F, 670, cy, 250, CH, "FUSE", EDGE, "merge by angle",
          ["a target in a seam is seen", "twice, by two cameras, and",
           "both are right &#8212; it is still", "one drone"])

    x, w = 960, 348
    F.svg.append(f'<rect x="{x}" y="{cy}" width="{w}" height="{CH}" rx="10" fill="{CARD}" '
                 f'stroke="{GREEN}" stroke-width="3"/>')
    F.header(x, cy, w, "DISCRIMINATE &#183; THE HARD PART", GREEN)
    text(F, x + w / 2, cy + 58, "watched flying", size=16.5, weight=700)
    block(F, x + w / 2, cy + 84,
          ["a rendered sky returns ~50 motion",
           "contacts a frame and no confidence",
           "threshold ranks the drone first.",
           "Physics does: a fixed object has a",
           "bearing rate of exactly zero.",
           ("85% of clutter survives the best one-frame gate", 600, GREEN)])

    panel(F, 1348, cy, 272, CH, "APPEARANCE &#183; DISPOSES", EDGE, "YOLO on a crop",
          ["640 px, aimed by motion,", "at native scale &#8212; a whole-",
           "frame pass would shrink a", "9 px drone to 8",
           ("and only it may set the range", 600, TXT)])
    panel(F, 1660, cy, 336, CH, "THE INTERFACE", BLUE, "TargetEstimate",
          ["a direction in body axes", "a pixel span", "a validity flag",
           "&#8212; and nothing else", ("guidance never sees an image", 600, TXT)])
    flow(F, boxes, cy + 98)

    text(F, W / 2, LY + LH - 22,
         "Motion proposes, appearance disposes. Every metre of detection range buys "
         "0.6 m of defended radius &#8212; nothing else in the system trades that steeply.",
         size=14, fill=SUB)

    # =======================================================================
    # 3 -- guidance
    # =======================================================================
    LY, LH = 808, 268
    lane(F, LY, LH, "3", "HIT &#8212; a bearing alone is enough to collide with something",
         "simulated flight (Isaac Sim, 20 Hz closed loop)", AMBER)
    cy = LY + 64
    CH = 158

    x, w = 44, 466
    F.svg.append(f'<rect x="{x}" y="{cy}" width="{w}" height="{CH}" rx="10" fill="{CARD}" '
                 f'stroke="{EDGE}" stroke-width="2"/>')
    F.header(x, cy, w, "MISSION STATE", EDGE)
    for i, (s, d) in enumerate([("SEARCH", "watch; do not move"),
                                ("ACQUIRE", "hold station until proven"),
                                ("PURSUE", "close under PN"),
                                ("TERMINAL", "commit")]):
        yy = cy + 56 + i * 25
        text(F, x + 24, yy, s, size=14, weight=700, fill=PURPLE, anchor="start")
        text(F, x + 132, yy, d, size=13.5, anchor="start", op=0.85)
    footnote(F, x, cy + CH + 22, w,
             "ACQUIRE holds still on purpose &#8212; moving destroys the evidence")

    x, w = 550, 466
    F.svg.append(f'<rect x="{x}" y="{cy}" width="{w}" height="{CH}" rx="10" fill="{CARD}" '
                 f'stroke="{PURPLE}" stroke-width="3"/>')
    F.header(x, cy, w, "THE CLOSURE LAW", PURPLE)
    text(F, x + w / 2, cy + 58, "proportional navigation", size=16.5, weight=700)
    F.svg.append(f'<rect x="{x + 70}" y="{cy + 72}" width="{w - 140}" height="34" rx="7" '
                 f'fill="{INNER}" stroke="{EDGE}"/>')
    text(F, x + w / 2, cy + 95, "a&#8869;  =  N &#183; Vc &#183; d&#955;/dt", size=16,
         fill=CYA, mono=True)
    block(F, x + w / 2, cy + 128,
          ["a line of sight that does not rotate while",
           "the range shrinks is a collision course"])
    footnote(F, x, cy + CH + 22, w,
             "bearing steers (a pixel is a ray); range only schedules speed")

    panel(F, 1056, cy, 320, CH, "AIRFRAME", EDGE, "body command",
          ["20 Hz; speed, acceleration", "and yaw rate all limited",
           "bearings stamped at capture"])
    footnote(F, 1056, cy + CH + 22, 320, "sensor latency is declared, not inherited")

    panel(F, 1416, cy, 580, CH, "RESULT &#183; CITY DEFENCE", BLUE,
          "24 / 24 intercepted &#183; 0 buildings hit",
          [("mean true closest approach 0.080 m &#8212; a sixth of the airframe", 600, TXT),
           "median 4.29 s to spare before the strike would have landed",
           "24 arrival bearings, one every 15&#176; of the compass"])
    flow(F, [(44, 466), (550, 466), (1056, 320), (1416, 580)], cy + 79)

    # =======================================================================
    # 4 -- how it is scored
    # =======================================================================
    LY, LH = 1104, 194
    lane(F, LY, LH, "4", "HOW EVERY NUMBER HERE WAS MEASURED",
         "no claim without an artifact", BLUE)
    cy = LY + 58
    panel(F, 44, cy, 620, 122, "DETECTION &#183; REAL VIDEO", GREEN,
          "centre distance, &#964; = 12 px",
          ["IoU is meaningless on a 6 px box &#8212; a pixel of shift swings it wildly",
           "scored on a video never trained on and never used to pick a model"],
          title_dy=52, lines_dy=76, lh=19)
    panel(F, 700, cy, 620, 122, "INTERCEPTION &#183; ISAAC SIM", AMBER,
          "true closest approach &lt; 1.0 m",
          ["two Iris airframes touching are ~0.5 m centre to centre",
           "arriving after the building is hit is a separate, scored failure"],
          title_dy=52, lines_dy=76, lh=19)
    panel(F, 1356, cy, 640, 122, "THE HARNESSES", EDGE, "two loops and a test suite",
          ["Isaac Sim at 20 Hz, and an arithmetic sandbox: 120 scenarios in 1.2 s",
           "540 unit tests over geometry, guidance, dynamics and the ring, in 22 s"],
          title_dy=52, lines_dy=76, lh=19)

    text(F, W / 2, H - 16,
         "Detection is measured on real, hand-labelled video. Interception is measured in a "
         "closed-loop Isaac Sim renderer &#8212; there is no flight test here, and this figure "
         "says so on purpose.", size=13, fill=SUB)

    F.write("architecture_system")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
