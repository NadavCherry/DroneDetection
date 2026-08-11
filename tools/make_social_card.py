#!/usr/bin/env python3
"""The 1280x640 social preview card.

Two places want one image that says what this is: GitHub's repository social
preview (Settings -> General -> Social preview) and the ``og:image`` a link
unfurls to on LinkedIn or Slack. Both crop toward the centre and both render it
small, so this is deliberately four numbers and a sentence rather than a
diagram -- the end-to-end figure is `make_arch_figure_system.py` and does not
survive being 400 px wide in a feed.

    .venv/bin/python tools/make_social_card.py
        -> docs/media/social_card.png / .svg

The provenance line is not decoration. It is the first thing a sceptical reader
looks for and the last thing a headline number usually says.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_arch_figures_final import BLUE, CARD, EDGE, GREEN, SUB, TXT, Fig  # noqa: E402

AMBER = "#d29922"
W, H = 1280, 640

STATS = [
    ("1.000", "AP / F1, detection", "unseen real video", GREEN),
    ("0.83", "mAP50, temporal stack", "0.06 from a single frame", GREEN),
    ("24 / 24", "intruders intercepted", "0 buildings hit", AMBER),
    ("0.080 m", "mean closest approach", "airframe span is 0.47 m", AMBER),
]


def main() -> int:
    F = Fig(W, H)

    F.svg.append(f'<text x="64" y="118" font-size="58" font-weight="700" fill="{TXT}">'
                 f'See the drone, then hit it</text>')
    F.svg.append(f'<text x="64" y="166" font-size="23" fill="{SUB}">'
                 f'Finding a drone <tspan font-weight="700" fill="{TXT}">3&#8211;14 pixels</tspan> '
                 f'wide in 720p video from a moving camera &#8212;</text>')
    F.svg.append(f'<text x="64" y="200" font-size="23" fill="{SUB}">'
                 f'then flying into it, with nothing but that camera.</text>')

    # four numbers, evenly spaced
    x0, gap, cw, ch = 64, 24, (W - 128 - 3 * 24) / 4, 190
    for i, (v, k, n, col) in enumerate(STATS):
        x = x0 + i * (cw + gap)
        y = 250
        F.svg.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" rx="14" '
                     f'fill="{CARD}" stroke="{EDGE}" stroke-width="2"/>')
        F.svg.append(f'<rect x="{x}" y="{y}" width="{cw}" height="5" rx="2.5" fill="{col}"/>')
        F.svg.append(f'<text x="{x + cw / 2}" y="{y + 76}" text-anchor="middle" font-size="42" '
                     f'font-weight="700" fill="{TXT}">{v}</text>')
        F.svg.append(f'<text x="{x + cw / 2}" y="{y + 112}" text-anchor="middle" font-size="17" '
                     f'font-weight="600" fill="{TXT}" opacity="0.9">{k}</text>')
        F.svg.append(f'<text x="{x + cw / 2}" y="{y + 140}" text-anchor="middle" font-size="14.5" '
                     f'fill="{SUB}">{n}</text>')

    # the provenance line -- the point of the two colours above
    y = 500
    F.svg.append(f'<circle cx="74" cy="{y - 5}" r="7" fill="{GREEN}"/>')
    F.svg.append(f'<text x="92" y="{y}" font-size="18" fill="{TXT}" opacity="0.9">'
                 f'detection measured on real hand-labelled video</text>')
    F.svg.append(f'<circle cx="622" cy="{y - 5}" r="7" fill="{AMBER}"/>')
    F.svg.append(f'<text x="640" y="{y}" font-size="18" fill="{TXT}" opacity="0.9">'
                 f'interception measured closed-loop in Isaac Sim &#8212; no flight test</text>')

    F.svg.append(f'<rect x="64" y="{y + 34}" width="{W - 128}" height="1.5" fill="{EDGE}"/>')
    F.svg.append(f'<text x="64" y="{y + 84}" font-size="20" font-weight="700" fill="{BLUE}">'
                 f'github.com/NadavCherry/SpeckLock</text>')
    F.svg.append(f'<text x="{W - 64}" y="{y + 84}" text-anchor="end" font-size="18" fill="{SUB}">'
                 f'Nadav Cherry</text>')

    F.write("social_card")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
