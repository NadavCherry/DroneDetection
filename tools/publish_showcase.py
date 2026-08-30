#!/usr/bin/env python3
"""Re-encode the curated showcase clips into ``docs/media/`` for the project site.

The in-loop recorders write OpenCV ``mp4v`` at 7-10 Mb/s, which is fine on disk
and hopeless on a web page: one 7-second engagement is 8 MB. Everything a
visitor sees is re-encoded here to H.264 at a web-sane bitrate, together with a
JPEG poster frame, and the facts for each clip are read *from the run's own
``results.json``* rather than typed in -- so a caption can never drift away from
the run it describes.

    .venv/bin/python tools/publish_showcase.py            # everything
    .venv/bin/python tools/publish_showcase.py --only city
    .venv/bin/python tools/publish_showcase.py --dry-run

Outputs
    docs/media/pursuit/city/<name>.mp4  + .jpg     the ten recorded ring engagements
    docs/media/pursuit/chase/<name>.mp4 + .jpg     six one-camera pursuits
    docs/media/showcase.json                       manifest the site reads

Two things worth keeping right, both learned the hard way (see also
``pursuit/tools/publish_clip.py``):

* **Speeding a clip up must drop frames, not relabel them.** ``setpts`` alone
  leaves a frame rate some players ignore; ``fps=`` after it makes the change real.
* **The target is 3 px of low contrast.** Encode ``yuv420p`` at a conservative
  CRF and scale with lanczos -- an aggressive CRF eats exactly the signal the
  whole project is about, and the clip then proves nothing.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "media"


# ---------------------------------------------------------------------------
# the curated set
# ---------------------------------------------------------------------------

@dataclass
class Clip:
    src: str                     # relative to repo root
    out: str                     # relative to docs/media, without extension
    results: str                 # results.json holding this engagement
    key: str                     # its `name` inside that results.json
    speed: float = 2.0
    width: int = 960
    crf: int = 22
    poster_at: float = 0.55       # fraction of the clip the poster frame comes from
    title: str = ""
    blurb: str = ""
    tags: list = field(default_factory=list)


CITY = [
    Clip("work/pursuit/city_clips/city-000.mp4", "pursuit/city/city-000",
         "work/pursuit/city_clips/results.json", "city-000",
         title="Due north, 0&deg;",
         blurb="Straight over the nose. Acquired in 0.20 s, intercepted at 6.0 s.",
         tags=["ring", "hit"]),
    Clip("work/pursuit/city_clips/city-030.mp4", "pursuit/city/city-030",
         "work/pursuit/city_clips/results.json", "city-030",
         title="030&deg; &mdash; off the bow",
         blurb="Outside a single camera's 76&deg; cone; the ring never has to turn.",
         tags=["ring", "hit"]),
    Clip("work/pursuit/city_clips/city-075.mp4", "pursuit/city/city-075",
         "work/pursuit/city_clips/results.json", "city-075",
         title="075&deg; &mdash; near a seam",
         blurb="Arrives close to the forward/right seam and is handed between cameras mid-track.",
         tags=["ring", "hit", "seam"]),
    Clip("work/pursuit/city_clips/city-105.mp4", "pursuit/city/city-105",
         "work/pursuit/city_clips/results.json", "city-105",
         title="105&deg; &mdash; the headline engagement",
         blurb="170 m out and 2.7 px across, diving on a plaza block. Passed 0.13 m from it.",
         tags=["ring", "hit", "headline"]),
    Clip("work/pursuit/city_clips/city-150.mp4", "pursuit/city/city-150",
         "work/pursuit/city_clips/results.json", "city-150",
         title="150&deg; &mdash; over the shoulder",
         blurb="Well behind the beam. A forward camera would still be slewing.",
         tags=["ring", "hit"]),
    Clip("work/pursuit/city_clips/city-180.mp4", "pursuit/city/city-180",
         "work/pursuit/city_clips/results.json", "city-180",
         title="180&deg; &mdash; dead astern",
         blurb="The exact bearing one nose camera cannot see. Acquired in 0.20 s.",
         tags=["ring", "hit", "astern"]),
    Clip("work/pursuit/city_clips/city-210.mp4", "pursuit/city/city-210",
         "work/pursuit/city_clips/results.json", "city-210",
         title="210&deg; &mdash; astern quarter",
         blurb="Closest approach 8 mm &mdash; about a sixtieth of the airframe's rotor span.",
         tags=["ring", "hit"]),
    Clip("work/pursuit/city_clips/city-255.mp4", "pursuit/city/city-255",
         "work/pursuit/city_clips/results.json", "city-255",
         title="255&deg; &mdash; from the port quarter",
         blurb="Aft camera first, then two seam crossings into the chase.",
         tags=["ring", "hit", "seam"]),
    Clip("work/pursuit/city_clips/city-285.mp4", "pursuit/city/city-285",
         "work/pursuit/city_clips/results.json", "city-285",
         title="285&deg; &mdash; abeam to port",
         blurb="90&deg; off the nose: the worst case for a single sensor, ordinary for a ring.",
         tags=["ring", "hit"]),
    Clip("work/pursuit/city_clips/city-330.mp4", "pursuit/city/city-330",
         "work/pursuit/city_clips/results.json", "city-330",
         title="330&deg; &mdash; closing the circle",
         blurb="Tenth of ten recorded arrivals; the full 24 cover every 15&deg;.",
         tags=["ring", "hit"]),
]

CHASE = [
    Clip("work/pursuit/final/skydome/in-crossing.mp4", "pursuit/chase/skydome-in-crossing",
         "work/pursuit/final/skydome/results.json", "in-crossing", speed=1.5, width=900,
         title="Best pass of the campaign &mdash; 2.5 cm",
         blurb="A crossing target at 115 m, open sky. Proportional navigation drives the "
               "line-of-sight rate to zero and the closure does the rest.",
         tags=["chase", "skydome", "hit"]),
    Clip("work/pursuit/final/rivermark/in-high-left.mp4", "pursuit/chase/rivermark-in-high-left",
         "work/pursuit/final/rivermark/results.json", "in-high-left", speed=1.5, width=900,
         title="Urban clutter, 115 m, high on the left",
         blurb="Rivermark rooftops behind the target for the whole run. 0.114 m.",
         tags=["chase", "rivermark", "hit"]),
    Clip("work/pursuit/final/rivermark/L5-break_turn.mp4", "pursuit/chase/rivermark-break-turn",
         "work/pursuit/final/rivermark/results.json", "L5-break_turn", speed=1.0, width=900,
         title="It breaks; the law does not care",
         blurb="A hard break turn at close range. PN is driven by bearing rate, so a manoeuvre "
               "is just a new rate to null.",
         tags=["chase", "rivermark", "hit", "evasion"]),
    Clip("work/pursuit/final/rivermark/outbound-60m.mp4", "pursuit/chase/rivermark-outbound-60m",
         "work/pursuit/final/rivermark/results.json", "outbound-60m", speed=1.5, width=900,
         title="A fleeing target has no bearing rate",
         blurb="Running straight away, which is the geometry a constant-rate test cannot help "
               "with &mdash; and the reason that test is only demanded of motion contacts.",
         tags=["chase", "rivermark", "hit"]),
    Clip("work/pursuit/final/skydome/in-behind.mp4", "pursuit/chase/skydome-in-behind",
         "work/pursuit/final/skydome/results.json", "in-behind", speed=3.0, width=900,
         title="Why the ring exists",
         blurb="One forward camera, target arriving from behind: 21.0 s to acquire and 30.5 s to "
               "intercept, nearly all of it spent turning to look. The ring does this in 0.2 s.",
         tags=["chase", "skydome", "hit", "slow"]),
    Clip("work/pursuit/final/rivermark/right-to-left-45m.mp4", "pursuit/chase/rivermark-miss",
         "work/pursuit/final/rivermark/results.json", "right-to-left-45m", speed=3.0, width=900,
         title="A failure, shown in full",
         blurb="Detection rate 0.00: the target crosses low against Rivermark's roofline and is "
               "never acquired, so there is nothing to steer at. Every one of the eight failures "
               "looks like this &mdash; perception, not guidance.",
         tags=["chase", "rivermark", "miss"]),
]


# ---------------------------------------------------------------------------

def ffmpeg() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def run(args: list[str]) -> None:
    proc = subprocess.run([str(a) for a in args], capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-2500:])
        raise SystemExit(f"ffmpeg failed: {' '.join(str(a) for a in args[:6])} ...")


def load_facts(results: Path, key: str) -> dict:
    """Pull one engagement's measured facts straight out of its run manifest.

    ``detector`` is carried through with the rest, and it is not decoration. The city
    clips were flown with ``detector: "oracle"`` -- the simulator's own bounding box,
    zero latency -- so their "detection rate 0.97" is that oracle's *visibility*, not a
    seeker's performance. The chase clips ran on a real trained detector. Both sets used
    to render identical captions, giving a reader no way to tell which was which, while
    the same city mission on the real detector scores 0.022-0.044.

    Reading it from the run manifest rather than labelling the cards by hand is the same
    rule the rest of this file follows: a caption that is typed can drift, a caption that
    is read cannot.
    """
    if not results.is_file():
        return {}
    blob = json.loads(results.read_text(encoding="utf-8"))
    detector = (blob.get("args") or {}).get("detector")
    for r in blob.get("results", []):
        if r.get("name") == key:
            return {
                "outcome": r.get("outcome"),
                "cpa_m": r.get("pass_cpa_m"),
                "t_intercept_s": r.get("time_to_intercept_s"),
                "acquire_s": r.get("acquire_time_s"),
                "margin_s": r.get("strike_margin_s"),
                "detect_rate": r.get("detect_rate"),
                "track_rate": r.get("track_rate"),
                "policy": r.get("policy"),
                "struck_asset": r.get("struck_asset"),
                # Per-engagement if the run recorded it, else the run-level setting.
                "detector": r.get("detector", detector),
            }
    return {}


def duration_s(path: Path, fx: str) -> float:
    """Clip length, read back off the encoded file (ffprobe is not shipped here)."""
    proc = subprocess.run([fx, "-hide_banner", "-i", str(path)],
                          capture_output=True, text=True)
    for line in proc.stderr.splitlines():
        if "Duration:" in line:
            hh, mm, ss = line.split("Duration:")[1].split(",")[0].strip().split(":")
            return int(hh) * 3600 + int(mm) * 60 + float(ss)
    return 0.0


def encode(clip: Clip, fx: str, dry: bool) -> dict | None:
    src = ROOT / clip.src
    if not src.is_file():
        print(f"  !! missing source, skipped: {clip.src}")
        return None
    dst = OUT / f"{clip.out}.mp4"
    poster = OUT / f"{clip.out}.jpg"
    dst.parent.mkdir(parents=True, exist_ok=True)

    # setpts retimes; the fps filter after it actually drops the frames.
    vf = (f"setpts={1.0 / clip.speed:.5f}*PTS,fps=25,"
          f"scale={clip.width}:-2:flags=lanczos")
    if not dry:
        run([fx, "-y", "-i", src, "-vf", vf, "-an",
             "-c:v", "libx264", "-preset", "slow", "-crf", clip.crf,
             "-profile:v", "high", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", dst])
        # Poster from ~55% in. Frame 0 of every one of these is the pre-roll
        # SEARCH state -- an empty sky and a HUD reading "target not in frame",
        # which is the least representative image in the clip.
        at = max(0.0, duration_s(dst, fx) * clip.poster_at)
        run([fx, "-y", "-ss", f"{at:.2f}", "-i", dst, "-frames:v", "1",
             "-q:v", "4", poster])

    size = dst.stat().st_size if dst.is_file() else 0
    print(f"  {clip.out:38s} {size / 1e6:5.2f} MB   (from {src.stat().st_size / 1e6:5.1f} MB)")
    return {
        "id": Path(clip.out).name,
        "mp4": f"{clip.out}.mp4",
        "poster": f"{clip.out}.jpg",
        "title": clip.title,
        "blurb": clip.blurb,
        "tags": clip.tags,
        "speed": clip.speed,
        "bytes": size,
        "source_clip": clip.src,
        "facts": load_facts(ROOT / clip.results, clip.key),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["city", "chase"], help="one group only")
    ap.add_argument("--dry-run", action="store_true", help="list what would be written")
    ap.add_argument("--facts-only", action="store_true",
                    help="re-read each clip's facts from its run manifest and rewrite "
                         "showcase.json, without re-encoding any video")
    a = ap.parse_args(argv)

    ### Facts change when a run is re-scored or a field is added; the video does not.
    ### Re-encoding 16 clips to correct a caption needs ffmpeg and the source recordings,
    ### which is why the `detector` field went missing from every card for so long -- the
    ### only way to refresh a caption was a job nobody wanted to run.
    if a.facts_only:
        path = OUT / "showcase.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        by_id = {c.out.rsplit("/", 1)[-1]: c for c in (*CITY, *CHASE)}
        changed = 0
        for group in manifest.values():
            for entry in group:
                clip = by_id.get(entry.get("id"))
                if clip is None:
                    print(f"  no clip definition for {entry.get('id')!r} -- left alone")
                    continue
                fresh = load_facts(ROOT / clip.results, clip.key)
                if fresh and fresh != entry.get("facts"):
                    entry["facts"] = fresh
                    changed += 1
        path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
        print(f"refreshed facts for {changed} clip(s) -> {path}")
        return 0

    fx = ffmpeg()
    groups = {"city": CITY, "chase": CHASE}
    if a.only:
        groups = {a.only: groups[a.only]}

    manifest: dict = {}
    total = 0
    for name, clips in groups.items():
        print(f"[{name}]")
        entries = [e for e in (encode(c, fx, a.dry_run) for c in clips) if e]
        manifest[name] = entries
        total += sum(e["bytes"] for e in entries)

    ### REFUSE to write a manifest that could not be built.
    ###
    ### The 16 source clips are Isaac Sim recordings under work/pursuit/, excluded by
    ### .gitignore and absent from every clone. encode() SKIPS a missing source and
    ### returns None; those Nones are filtered out, and the surviving EMPTY list was then
    ### merged over the tracked manifest -- emptying docs/media/showcase.json, after which
    ### tools/make_gallery.py (the very next line of the README's reproduction block)
    ### rewrote docs/gallery.html from nothing. Two tracked files destroyed, silently, by
    ### following the documented instructions.
    ###
    ### An empty group is never a legitimate result: every group is non-empty by
    ### construction, so zero entries means zero sources were found.
    empty = [name for name, entries in manifest.items() if not entries]
    if empty and not a.dry_run:
        raise SystemExit(
            "no source clips found for: " + ", ".join(empty)
            + ". These are Isaac Sim recordings under work/pursuit/ and are NOT in "
            "the repository (.gitignore). Nothing was written; "
            "docs/media/showcase.json is unchanged. To rebuild the gallery from the "
            "tracked manifest run tools/make_gallery.py on its own -- this script is "
            "for the authors, who have the recordings.")

    if not a.dry_run:
        path = OUT / "showcase.json"
        merged = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        merged.update(manifest)
        path.write_text(json.dumps(merged, indent=1) + "\n", encoding="utf-8")
        print(f"\nwrote {path}  --  {total / 1e6:.1f} MB of video in this pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
