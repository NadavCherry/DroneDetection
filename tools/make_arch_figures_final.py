"""Generate the final-profile architecture figures (same visual language as
tools/make_arch_figure.py): docs/media/architecture_pcmax.svg/.png and
docs/media/architecture_edgert.svg/.png."""

from __future__ import annotations

from pathlib import Path

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


class Fig:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" font-family="Segoe UI, Helvetica, Arial, sans-serif">',
            f'<rect width="{w}" height="{h}" fill="{BG}" rx="12"/>',
            f'''<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5"
             markerWidth="7" markerHeight="7" orient="auto-start-reverse">
             <path d="M 0 1 L 9 5 L 0 9 z" fill="{ARROW}"/></marker></defs>''']

    def header(self, x, y, w, text, color):
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="30" rx="10" fill="{color}"/>')
        self.svg.append(f'<rect x="{x}" y="{y+18}" width="{w}" height="12" fill="{color}"/>')
        self.svg.append(f'<text x="{x+w/2}" y="{y+20}" text-anchor="middle" font-size="13.5" '
                        f'font-weight="700" fill="#ffffff" letter-spacing="0.8">{text}</text>')

    def card(self, x, y, w, h, badge, badge_color, title, lines, sub=None, accent=None,
             accent_w=2):
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" '
                        f'fill="{CARD}" stroke="{accent or EDGE}" stroke-width="{accent_w}"/>')
        self.header(x, y, w, badge, badge_color)
        ty = y + 56
        if title:
            self.svg.append(f'<text x="{x+w/2}" y="{ty}" text-anchor="middle" font-size="16.5" '
                            f'font-weight="700" fill="{TXT}">{title}</text>')
            ty += 24
        for i, ln in enumerate(lines):
            self.svg.append(f'<text x="{x+w/2}" y="{ty+i*20}" text-anchor="middle" '
                            f'font-size="13.5" fill="{TXT}" opacity="0.87">{ln}</text>')
        if sub:
            self.svg.append(f'<text x="{x+w/2}" y="{y+h+21}" text-anchor="middle" '
                            f'font-size="12.5" fill="{SUB}">{sub}</text>')

    def arrow(self, pts, label=None, loff=(0, -9)):
        d = "M " + " L ".join(f"{px} {py}" for px, py in pts)
        self.svg.append(f'<path d="{d}" fill="none" stroke="{ARROW}" stroke-width="2.4" '
                        f'marker-end="url(#ar)"/>')
        if label:
            mx, my = pts[len(pts) // 2]
            self.svg.append(f'<text x="{mx+loff[0]}" y="{my+loff[1]}" text-anchor="middle" '
                            f'font-size="12.5" fill="{SUB}">{label}</text>')

    def squares_motif(self, x, y):
        """The three-moments-one-image motif (t-12 / t-6 / now)."""
        for off, c in [(0, YEL), (16, MAG), (32, CYA)]:
            self.svg.append(f'<rect x="{x+off}" y="{y+off*0.55}" width="52" height="52" rx="7" '
                            f'fill="none" stroke="{c}" stroke-width="3.5"/>')

    def corner_tag(self, title, sub):
        self.svg.append(f'<text x="{self.w-30}" y="40" text-anchor="end" font-size="19" '
                        f'font-weight="700" fill="{TXT}">{title}</text>')
        self.svg.append(f'<text x="{self.w-30}" y="60" text-anchor="end" font-size="12.5" '
                        f'fill="{SUB}">{sub}</text>')

    def footer(self, text):
        self.svg.append(f'<text x="{self.w/2}" y="{self.h-16}" text-anchor="middle" '
                        f'font-size="13" fill="{SUB}">{text}</text>')

    def write(self, stem):
        out = "\n".join(self.svg) + "\n</svg>"
        Path("docs/media").mkdir(parents=True, exist_ok=True)
        Path(f"docs/media/{stem}.svg").write_text(out)
        import cairosvg

        cairosvg.svg2png(bytestring=out.encode(), write_to=f"docs/media/{stem}.png",
                         output_width=self.w)
        print(f"written docs/media/{stem}.svg/.png")


# ===========================================================================
# PC-MAX: three detection streams -> fusion -> tracker -> track classifier
# ===========================================================================

def make_pcmax():
    F = Fig(1840, 800)
    F.corner_tag("PC-MAX", "accuracy-first desktop profile &#183; final/run_final.py --profile pc-max")
    TOP = 70          # lane 1: find + identify movers
    MID = 350         # lane 2: the two full-frame streams
    BOT = 570         # lane 3: fuse -> track -> classify -> output

    # ---- lane 1 -----------------------------------------------------------
    F.card(30, TOP + 20, 170, 130, "INPUT", BLUE, "frame t",
           ["1280 &#215; 720 RGB", "30 fps video"])
    F.card(280, TOP + 20, 230, 130, "STAGE 0", EDGE, "stabilize",
           ["camera motion removed", "(aligned to frame 0)"],
           sub="phase correlation &#183; keeps a buffer of aligned grays")

    # proposals card with two inner channels (same stage as the HiveLab figure)
    px, py, pw, ph = 590, TOP, 350, 190
    F.svg.append(f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" rx="10" '
                 f'fill="{CARD}" stroke="{EDGE}" stroke-width="2"/>')
    F.header(px, py, pw, "STAGE 1 &#183; MOTION PROPOSALS", EDGE)
    for i, (t, s) in enumerate([
            ("slow-mover channel", "lagged background &#8212; catches 0.5 px/frame drifters"),
            ("fast channel (MOG2)", "maximum recall &#8212; precision comes later")]):
        yy = py + 44 + i * 62
        F.svg.append(f'<rect x="{px+18}" y="{yy}" width="{pw-36}" height="54" rx="7" '
                     f'fill="{INNER}" stroke="{EDGE}"/>')
        F.svg.append(f'<text x="{px+pw/2}" y="{yy+23}" text-anchor="middle" font-size="14.5" '
                     f'font-weight="600" fill="{TXT}">{t}</text>')
        F.svg.append(f'<text x="{px+pw/2}" y="{yy+43}" text-anchor="middle" font-size="12.5" '
                     f'fill="{SUB}">{s}</text>')
    F.svg.append(f'<text x="{px+pw/2}" y="{py+ph+21}" text-anchor="middle" font-size="12.5" '
                 f'fill="{SUB}">union of both &#8250; top-20 moving candidates per frame</text>')

    # temporal verifier
    vx, vy, vw, vh = 1020, TOP, 400, 190
    F.svg.append(f'<rect x="{vx}" y="{vy}" width="{vw}" height="{vh}" rx="10" '
                 f'fill="{CARD}" stroke="{GREEN}" stroke-width="3"/>')
    F.header(vx, vy, vw, "STAGE 2 &#183; TEMPORAL VERIFIER", GREEN)
    F.squares_motif(vx + 34, vy + 56)
    F.svg.append(f'<text x="{vx+255}" y="{vy+70}" text-anchor="middle" font-size="15.5" '
                 f'font-weight="700" fill="{TXT}">three moments, one image</text>')
    F.svg.append(f'<text x="{vx+165}" y="{vy+97}" font-size="14.5" font-weight="700" fill="{YEL}">t-12</text>')
    F.svg.append(f'<text x="{vx+205}" y="{vy+97}" font-size="14.5" font-weight="700" fill="{MAG}">t-6</text>')
    F.svg.append(f'<text x="{vx+238}" y="{vy+97}" font-size="14.5" font-weight="700" fill="{CYA}">now</text>')
    F.svg.append(f'<text x="{vx+278}" y="{vy+97}" font-size="13.5" fill="{TXT}" opacity="0.85">= color channels</text>')
    F.svg.append(f'<text x="{vx+vw/2}" y="{vy+136}" text-anchor="middle" font-size="14" '
                 f'fill="{TXT}" opacity="0.9">yolov8s-P2 on 640&#178; crops at 1:1 scale (verifier640)</text>')
    F.svg.append(f'<text x="{vx+vw/2}" y="{vy+160}" text-anchor="middle" font-size="15" '
                 f'font-weight="700" fill="{TXT}">&#8250; drone / bird / nothing</text>')
    F.svg.append(f'<text x="{vx+vw/2}" y="{vy+vh+21}" text-anchor="middle" font-size="12.5" '
                 f'fill="{SUB}">static world cancels to gray &#8212; only movers get color</text>')

    # ---- lane 2: the two extra streams -------------------------------------
    F.card(30, MID, 230, 140, "STAGE 3", EDGE, "near/big expert",
           ["separate YOLO @ 1280", "on the raw RGB frame:", "large / hovering / landed"],
           sub="covers what motion cannot see")
    F.card(300, MID, 230, 140, "STAGE 4", EDGE, "full-frame temporal",
           ["yolov8s-P2 @ 1280 reads", "the whole 3-moment stack", "(fullS)"],
           sub="independent full-frame stream")

    # ---- lane 3: fuse -> track -> classify -> output -----------------------
    F.card(590, BOT, 190, 150, "STAGE 5", EDGE, "fuse",
           ["centers within 10 px", "merge into one detection", "with a noisy-OR score"],
           sub="agreement between streams is evidence")
    F.card(850, BOT, 280, 150, "STAGE 6 &#183; TRACKING", GREEN, "",
           ["Kalman tracker, camera-", "compensated; coasts through fades,",
            "re-locks with a strict local search"],
           sub="foliage jitters in place &#8212; real aircraft travel",
           accent=GREEN)
    F.card(1200, BOT, 300, 150, "STAGE 7 &#183; TRACK CLASSIFIER", GREEN, "",
           ["aggregated verifier verdicts:", "DRONE = 50%+ confirmed detections",
            "and 8+ of them (sustained evidence)"],
           sub="per-frame ambiguity &#8250; track-level certainty",
           accent=GREEN, accent_w=3)

    ox, oy, ow, oh = 1540, BOT - 40, 270, 200
    F.svg.append(f'<rect x="{ox}" y="{oy}" width="{ow}" height="{oh}" rx="10" '
                 f'fill="{CARD}" stroke="{BLUE}" stroke-width="2.5"/>')
    F.header(ox, oy, ow, "OUTPUT", BLUE)
    for i, (c, t, s) in enumerate([
            ("#f85149", "DRONE alarm", "alarm on the 8th confirmed hit"),
            ("#d29922", "near drone", "landed / hovering (big boxes)"),
            ("#6e7681", "discarded", "birds, foliage, unknown movers")]):
        yy = oy + 66 + i * 44
        F.svg.append(f'<circle cx="{ox+30}" cy="{yy-5}" r="8.5" fill="{c}"/>')
        F.svg.append(f'<text x="{ox+50}" y="{yy}" font-size="15.5" font-weight="700" '
                     f'fill="{TXT}">{t}</text>')
        F.svg.append(f'<text x="{ox+50}" y="{yy+19}" font-size="12.5" fill="{SUB}">{s}</text>')

    # ---- arrows -------------------------------------------------------------
    F.arrow([(200, TOP + 85), (278, TOP + 85)])
    F.arrow([(510, TOP + 85), (588, TOP + 85)], "aligned", loff=(0, -12))
    F.arrow([(940, TOP + 95), (1018, TOP + 95)], "movers", loff=(0, -12))
    # input -> expert (raw frame, down the left edge)
    F.arrow([(115, TOP + 150), (115, MID - 2)], "raw frame", loff=(-52, -60))
    # stabilize -> full-frame temporal (the same aligned-gray stack)
    F.arrow([(415, TOP + 150), (415, MID - 2)], "3-moment stack", loff=(62, -60))
    # expert -> fuse (around the bottom)
    F.arrow([(145, MID + 140), (145, BOT + 115), (588, BOT + 115)], "large / landed",
            loff=(-62, -30))
    # fullS -> fuse
    F.arrow([(415, MID + 140), (415, BOT + 45), (588, BOT + 45)], "full-frame dets",
            loff=(-64, -32))
    # verifier -> fuse (down and left, above lane 3)
    F.arrow([(1220, TOP + 190 + 30), (1220, BOT - 45), (685, BOT - 45), (685, BOT - 2)],
            "verified candidates", loff=(0, -10))
    # fuse -> tracker -> classifier -> output
    F.arrow([(780, BOT + 75), (848, BOT + 75)], "detections", loff=(0, -14))
    F.arrow([(1130, BOT + 75), (1198, BOT + 75)], "tracks", loff=(0, -14))
    F.arrow([(1500, BOT + 75), (1538, BOT + 75)])

    F.footer("unseen test video: tracked AP / F1 / R / P = 1.000, zero false alarms &#183; "
             "~4 fps on an RTX 5070 laptop &#8212; the accuracy-first profile")
    F.write("architecture_pcmax")


# ===========================================================================
# EDGE-RT: one nano network on the stabilized 3-frame stack, TensorRT FP16
# ===========================================================================

def make_edgert():
    F = Fig(1840, 620)
    F.corner_tag("EDGE-RT", "real-time edge profile &#183; final/run_final.py --profile edge-rt")
    TOP, BOT = 70, 400

    # ---- lane 1: frame -> stabilize -> stack -> one net ---------------------
    F.card(30, TOP + 30, 170, 130, "INPUT", BLUE, "frame t",
           ["1280 &#215; 720 BGR", "30 fps stream"])
    F.card(270, TOP + 10, 260, 170, "STAGE 0 &#183; LITE STABILIZER", EDGE, "stabilize (CPU)",
           ["phase-correlate a fixed", "768&#215;448 central crop", "against frame 0"],
           sub="full-frame precision at 1/3 the cost &#183; 4.3 ms")

    # temporal stack card (the same three-squares motif)
    sx, sy, sw, sh = 600, TOP, 330, 190
    F.svg.append(f'<rect x="{sx}" y="{sy}" width="{sw}" height="{sh}" rx="10" '
                 f'fill="{CARD}" stroke="{EDGE}" stroke-width="2"/>')
    F.header(sx, sy, sw, "STAGE 1 &#183; TEMPORAL STACK", EDGE)
    F.squares_motif(sx + 34, sy + 56)
    F.svg.append(f'<text x="{sx+215}" y="{sy+70}" text-anchor="middle" font-size="15.5" '
                 f'font-weight="700" fill="{TXT}">three moments,</text>')
    F.svg.append(f'<text x="{sx+215}" y="{sy+92}" text-anchor="middle" font-size="15.5" '
                 f'font-weight="700" fill="{TXT}">one image</text>')
    F.svg.append(f'<text x="{sx+142}" y="{sy+135}" font-size="14.5" font-weight="700" fill="{YEL}">t-12</text>')
    F.svg.append(f'<text x="{sx+182}" y="{sy+135}" font-size="14.5" font-weight="700" fill="{MAG}">t-6</text>')
    F.svg.append(f'<text x="{sx+215}" y="{sy+135}" font-size="14.5" font-weight="700" fill="{CYA}">now</text>')
    F.svg.append(f'<text x="{sx+257}" y="{sy+135}" font-size="13.5" fill="{TXT}" opacity="0.85">= channels</text>')
    F.svg.append(f'<text x="{sx+sw/2}" y="{sy+165}" text-anchor="middle" font-size="13.5" '
                 f'fill="{TXT}" opacity="0.87">stabilized grays from a 13-frame ring buffer</text>')
    F.svg.append(f'<text x="{sx+sw/2}" y="{sy+sh+21}" text-anchor="middle" font-size="12.5" '
                 f'fill="{SUB}">static world cancels to gray &#8212; movers leave a colored trail</text>')

    # the one nano net
    nx, ny, nw, nh = 1000, TOP, 400, 190
    F.svg.append(f'<rect x="{nx}" y="{ny}" width="{nw}" height="{nh}" rx="10" '
                 f'fill="{CARD}" stroke="{GREEN}" stroke-width="3"/>')
    F.header(nx, ny, nw, "STAGE 2 &#183; ONE NANO NETWORK", GREEN)
    F.svg.append(f'<text x="{nx+nw/2}" y="{ny+68}" text-anchor="middle" font-size="16.5" '
                 f'font-weight="700" fill="{TXT}">yolov8n-P2 @ 1280 &#8212; TensorRT FP16</text>')
    for i, ln in enumerate(["reads the whole stack, full frame",
                            "no proposals &#183; no crops &#183; no expert",
                            "&#8250; drone / bird, everywhere"]):
        F.svg.append(f'<text x="{nx+nw/2}" y="{ny+98+i*24}" text-anchor="middle" '
                     f'font-size="14" fill="{TXT}" opacity="0.9">{ln}</text>')
    F.svg.append(f'<text x="{nx+nw/2}" y="{ny+nh+21}" text-anchor="middle" font-size="12.5" '
                 f'fill="{SUB}">6.7 ms/frame &#8212; the entire neural budget</text>')

    # ---- lane 2: measured budget | tracker -> classifier -> output ----------
    bx, by, bw, bh = 30, BOT - 30, 480, 190
    F.svg.append(f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" rx="10" '
                 f'fill="{CARD}" stroke="{EDGE}" stroke-width="2"/>')
    F.header(bx, by, bw, "MEASURED &#183; RTX 5070, 591-FRAME TEST VIDEO", EDGE)
    rows = [("stabilize + warp", "4.3 ms"), ("nano net (TensorRT FP16)", "6.7 ms"),
            ("Kalman tracker", "0.1 ms"),
            ("end-to-end", "11 ms &#183; 74 fps")]
    for i, (k, v) in enumerate(rows):
        yy = by + 66 + i * 26
        w8 = "700" if i == len(rows) - 1 else "400"
        F.svg.append(f'<text x="{bx+28}" y="{yy}" font-size="13.5" fill="{TXT}" '
                     f'opacity="0.9" font-weight="{w8}">{k}</text>')
        F.svg.append(f'<text x="{bx+bw-28}" y="{yy}" text-anchor="end" font-size="13.5" '
                     f'font-weight="{w8}" fill="{TXT}">{v}</text>')
    F.svg.append(f'<text x="{bx+bw/2}" y="{by+bh+21}" text-anchor="middle" font-size="12.5" '
                 f'fill="{SUB}">projected on Jetson Orin Nano: 10-15 fps @1280, 25-35 fps @640</text>')

    F.card(640, BOT, 280, 140, "STAGE 3 &#183; TRACKING", GREEN, "",
           ["Kalman tracker, camera-", "compensated; coasting +", "strict local re-acquisition"],
           sub="same tracker as PC-MAX &#183; 0.1 ms", accent=GREEN)
    F.card(1000, BOT, 300, 140, "STAGE 4 &#183; TRACK CLASSIFIER", GREEN, "",
           ["DRONE = 50%+ confirmed", "detections and 8+ of them",
            "&#8212; alarm ~0.3 s after track birth"],
           sub="per-frame ambiguity &#8250; track-level certainty", accent=GREEN)

    ox, oy, ow, oh = 1480, BOT - 40, 330, 190
    F.svg.append(f'<rect x="{ox}" y="{oy}" width="{ow}" height="{oh}" rx="10" '
                 f'fill="{CARD}" stroke="{BLUE}" stroke-width="2.5"/>')
    F.header(ox, oy, ow, "OUTPUT", BLUE)
    for i, (c, t, s) in enumerate([
            ("#f85149", "DRONE alarm", "track confirmed by the net, repeatedly"),
            ("#6e7681", "discarded", "birds, foliage, unknown movers")]):
        yy = oy + 74 + i * 52
        F.svg.append(f'<circle cx="{ox+36}" cy="{yy-5}" r="8.5" fill="{c}"/>')
        F.svg.append(f'<text x="{ox+56}" y="{yy}" font-size="15.5" font-weight="700" '
                     f'fill="{TXT}">{t}</text>')
        F.svg.append(f'<text x="{ox+56}" y="{yy+19}" font-size="12.5" fill="{SUB}">{s}</text>')

    # ---- arrows -------------------------------------------------------------
    F.arrow([(200, TOP + 95), (268, TOP + 95)], "gray", loff=(0, -12))
    F.arrow([(530, TOP + 95), (598, TOP + 95)], "aligned", loff=(0, -12))
    F.arrow([(930, TOP + 95), (998, TOP + 95)], "1280&#215;720&#215;3", loff=(0, -12))
    # net -> tracker (down and left)
    F.arrow([(1200, TOP + 190 + 30), (1200, BOT - 55), (780, BOT - 55), (780, BOT - 2)],
            "detections, back in original coords", loff=(0, -10))
    F.arrow([(920, BOT + 70), (998, BOT + 70)], "tracks", loff=(0, -14))
    F.arrow([(1300, BOT + 70), (1478, BOT + 70)], "drone tracks only", loff=(0, -14))

    F.footer("unseen test video: tracked AP / F1 / R / P = 1.000, zero false alarms &#8212; "
             "at 74 fps end-to-end &#183; one network is the whole pipeline")
    F.write("architecture_edgert")


if __name__ == "__main__":
    make_pcmax()
    make_edgert()
