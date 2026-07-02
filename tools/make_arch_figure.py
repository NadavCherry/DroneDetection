"""Generate the README architecture figure (docs/media/architecture.svg/.png)."""

from __future__ import annotations

from pathlib import Path

W, H = 1840, 720
BG = "#0d1117"
CARD = "#161b22"
INNER = "#1c2430"
EDGE = "#30363d"
TXT = "#e6edf3"
SUB = "#9198a1"
BLUE = "#388bfd"
GREEN = "#2ea043"
YEL, MAG, CYA = "#e3b341", "#db61a2", "#39c5cf"
ARROW = "#8b949e"

svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
       f'viewBox="0 0 {W} {H}" font-family="Segoe UI, Helvetica, Arial, sans-serif">',
       f'<rect width="{W}" height="{H}" fill="{BG}" rx="12"/>',
       f'''<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5"
        markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M 0 1 L 9 5 L 0 9 z" fill="{ARROW}"/></marker></defs>''']


def header(x, y, w, text, color):
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="30" rx="10" fill="{color}"/>')
    svg.append(f'<rect x="{x}" y="{y+18}" width="{w}" height="12" fill="{color}"/>')
    svg.append(f'<text x="{x+w/2}" y="{y+20}" text-anchor="middle" font-size="13.5" '
               f'font-weight="700" fill="#ffffff" letter-spacing="0.8">{text}</text>')


def card(x, y, w, h, badge, badge_color, title, lines, sub=None, accent=None):
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" '
               f'fill="{CARD}" stroke="{accent or EDGE}" stroke-width="2"/>')
    header(x, y, w, badge, badge_color)
    ty = y + 56
    if title:
        svg.append(f'<text x="{x+w/2}" y="{ty}" text-anchor="middle" font-size="16.5" '
                   f'font-weight="700" fill="{TXT}">{title}</text>')
        ty += 24
    for i, ln in enumerate(lines):
        svg.append(f'<text x="{x+w/2}" y="{ty+i*20}" text-anchor="middle" '
                   f'font-size="13.5" fill="{TXT}" opacity="0.87">{ln}</text>')
    if sub:
        svg.append(f'<text x="{x+w/2}" y="{y+h+21}" text-anchor="middle" '
                   f'font-size="12.5" fill="{SUB}">{sub}</text>')


def arrow(pts, label=None, loff=(0, -9)):
    d = "M " + " L ".join(f"{px} {py}" for px, py in pts)
    svg.append(f'<path d="{d}" fill="none" stroke="{ARROW}" stroke-width="2.4" '
               f'marker-end="url(#ar)"/>')
    if label:
        mx, my = pts[len(pts) // 2]
        svg.append(f'<text x="{mx+loff[0]}" y="{my+loff[1]}" text-anchor="middle" '
                   f'font-size="12.5" fill="{SUB}">{label}</text>')


TOP, BOT = 70, 430  # lane y

# ---- top lane -------------------------------------------------------------
card(30, TOP + 20, 170, 130, "INPUT", BLUE, "frame t",
     ["1280 &#215; 720 RGB", "30 fps video"])

card(280, TOP + 20, 230, 130, "STAGE 0", EDGE, "stabilize",
     ["camera motion removed", "(aligned to frame 0)"],
     sub="phase correlation &#183; keeps a buffer of aligned grays")

# proposals card with two inner channels
px, py, pw, ph = 590, TOP, 350, 190
svg.append(f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" rx="10" '
           f'fill="{CARD}" stroke="{EDGE}" stroke-width="2"/>')
header(px, py, pw, "STAGE 1 &#183; MOTION PROPOSALS", EDGE)
for i, (t, s) in enumerate([
        ("slow-mover channel", "lagged background &#8212; catches 0.5 px/frame drifters"),
        ("fast channel (MOG2)", "maximum recall &#8212; precision comes later")]):
    yy = py + 44 + i * 62
    svg.append(f'<rect x="{px+18}" y="{yy}" width="{pw-36}" height="54" rx="7" '
               f'fill="{INNER}" stroke="{EDGE}"/>')
    svg.append(f'<text x="{px+pw/2}" y="{yy+23}" text-anchor="middle" font-size="14.5" '
               f'font-weight="600" fill="{TXT}">{t}</text>')
    svg.append(f'<text x="{px+pw/2}" y="{yy+43}" text-anchor="middle" font-size="12.5" '
               f'fill="{SUB}">{s}</text>')
svg.append(f'<text x="{px+pw/2}" y="{py+ph+21}" text-anchor="middle" font-size="12.5" '
           f'fill="{SUB}">union of both &#8250; top-20 moving candidates per frame</text>')

# temporal verifier (the star)
vx, vy, vw, vh = 1020, TOP, 400, 190
svg.append(f'<rect x="{vx}" y="{vy}" width="{vw}" height="{vh}" rx="10" '
           f'fill="{CARD}" stroke="{GREEN}" stroke-width="3"/>')
header(vx, vy, vw, "STAGE 2 &#183; TEMPORAL VERIFIER", GREEN)
for off, c in [(0, YEL), (16, MAG), (32, CYA)]:
    svg.append(f'<rect x="{vx+34+off}" y="{vy+56+off*0.55}" width="52" height="52" rx="7" '
               f'fill="none" stroke="{c}" stroke-width="3.5"/>')
svg.append(f'<text x="{vx+255}" y="{vy+70}" text-anchor="middle" font-size="15.5" '
           f'font-weight="700" fill="{TXT}">three moments, one image</text>')
svg.append(f'<text x="{vx+165}" y="{vy+97}" font-size="14.5" font-weight="700" fill="{YEL}">t-12</text>')
svg.append(f'<text x="{vx+205}" y="{vy+97}" font-size="14.5" font-weight="700" fill="{MAG}">t-6</text>')
svg.append(f'<text x="{vx+238}" y="{vy+97}" font-size="14.5" font-weight="700" fill="{CYA}">now</text>')
svg.append(f'<text x="{vx+278}" y="{vy+97}" font-size="13.5" fill="{TXT}" opacity="0.85">= color channels</text>')
svg.append(f'<text x="{vx+vw/2}" y="{vy+136}" text-anchor="middle" font-size="14" '
           f'fill="{TXT}" opacity="0.9">YOLO reads the motion trail of each candidate</text>')
svg.append(f'<text x="{vx+vw/2}" y="{vy+160}" text-anchor="middle" font-size="15" '
           f'font-weight="700" fill="{TXT}">&#8250; drone / bird / nothing</text>')
svg.append(f'<text x="{vx+vw/2}" y="{vy+vh+21}" text-anchor="middle" font-size="12.5" '
           f'fill="{SUB}">static world cancels to gray &#8212; only movers get color (see picture above)</text>')

# ---- bottom lane ----------------------------------------------------------
card(280, BOT, 230, 140, "STAGE 3", EDGE, "full-frame expert",
     ["separate YOLO @ 1280", "large / hovering /", "landed drones"],
     sub="covers what motion cannot see")

card(590, BOT, 230, 140, "STAGE 4", EDGE, "fuse",
     ["confirmed drone: high score", "bird: suppressed", "+ expert detections"],
     sub="one scored list per frame")

card(900, BOT, 330, 140, "STAGE 5 &#183; TRACKING", GREEN, "",
     ["Kalman tracker, camera-compensated", "coasts through fades, re-locks with",
      "a strict local search"],
     sub="foliage jitters in place &#8212; real aircraft travel &#8250; clutter tracks die",
     accent=GREEN)

ox, oy, ow, oh = 1490, BOT - 30, 320, 200
svg.append(f'<rect x="{ox}" y="{oy}" width="{ow}" height="{oh}" rx="10" '
           f'fill="{CARD}" stroke="{BLUE}" stroke-width="2.5"/>')
header(ox, oy, ow, "OUTPUT", BLUE)
for i, (c, t, s) in enumerate([
        ("#f85149", "DRONE alarm", "track confirmed by the verifier, repeatedly"),
        ("#d29922", "flying object", "directed mover, unconfirmed (birds)"),
        ("#6e7681", "discarded", "everything else never surfaces")]):
    yy = oy + 66 + i * 44
    svg.append(f'<circle cx="{ox+36}" cy="{yy-5}" r="8.5" fill="{c}"/>')
    svg.append(f'<text x="{ox+56}" y="{yy}" font-size="15.5" font-weight="700" '
               f'fill="{TXT}">{t}</text>')
    svg.append(f'<text x="{ox+56}" y="{yy+19}" font-size="12.5" fill="{SUB}">{s}</text>')

# ---- arrows ---------------------------------------------------------------
arrow([(200, TOP + 85), (278, TOP + 85)])
arrow([(510, TOP + 85), (588, TOP + 85)], "aligned", loff=(0, -12))
arrow([(940, TOP + 95), (1018, TOP + 95)], "movers", loff=(0, -12))
# input -> expert (down the left side)
arrow([(115, TOP + 150), (115, BOT + 70), (278, BOT + 70)], "raw frame", loff=(-60, -12))
# verifier -> fuse (down)
arrow([(1220, TOP + 190 + 30), (1220, BOT - 60), (705, BOT - 60), (705, BOT - 2)],
      "verified candidates", loff=(0, -10))
# expert -> fuse
arrow([(510, BOT + 70), (588, BOT + 70)])
# fuse -> tracker
arrow([(820, BOT + 70), (898, BOT + 70)], "detections", loff=(0, -14))
# tracker -> output
arrow([(1230, BOT + 70), (1488, BOT + 70)], "tracks")

Path("docs/media").mkdir(parents=True, exist_ok=True)
out = "\n".join(svg) + "\n</svg>"
Path("docs/media/architecture.svg").write_text(out)
import cairosvg

cairosvg.svg2png(bytestring=out.encode(), write_to="docs/media/architecture.png",
                 output_width=W)
print("written architecture.svg/.png")
