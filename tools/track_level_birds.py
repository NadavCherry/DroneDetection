#!/usr/bin/env python3
"""Bird rejection measured where the system actually decides: at the track.

WHY THIS EXISTS
---------------
Bird rejection is the hardest thing this pipeline does and the thing the project is sold
on, and until now every "track-level" bird number in this repository was a per-frame
number in disguise: `tools/tracks_to_dets.py` flattens tracks back into per-frame boxes
and `dronedet.metrics` scores those. That answers "how many frames contained a spurious
box", which is not the operational question. The operational question is **how many times
did the system raise a bird as a target**, and a bird raised for 150 consecutive frames is
ONE false alarm to an operator, not 150.

So this scores tracks as tracks. Nothing here is derived from a per-frame AP.

WHAT IS COUNTED, AND AGAINST WHAT
---------------------------------
07_05 carries eight hand-labelled bird tracks (934 boxes, median 6.0 px -- the same size
band as the 8.0 px drone) flagged ``ignore`` in the ground truth. `dronedet.metrics`
treats such objects as *distractors*: real things the detector must not call a drone. They
never contribute to recall, and a hit on one is neither a hit nor a miss on the target --
it is its own outcome. That convention is kept here and lifted to the track level.

Every confirmed track is assigned a ground-truth kind by where the plurality of its frames
sit -- ``target`` / ``bird`` / ``nothing`` -- and cross-tabulated against what the
classifier called it (``drone`` / ``near`` / ``other``). That table is the answer: the
cell [predicted drone x actually bird] is the bird false-alarm count, and it has never
been reported before.

THE CLAUSE MOST LIKELY TO LET A BIRD THROUGH
--------------------------------------------
`dronedet.trackclass` promotes a track to ``drone`` if ``confirmed OR sustained``, where
``sustained`` means ``n_tracked >= LONG_TRACK`` (120 frames) and carries **no appearance
requirement at all**. Its inline justification is that the tracker's directedness filter
has already removed clutter -- but `dronedet/track.py`'s own comment says the opposite:
"Birds pass too -- bird vs drone is an appearance/classifier problem, not a kinematic
one." A bird that flies straight for four seconds is exactly what that clause admits.
So every promotion is attributed to the clause that caused it, and tracks promoted by
``sustained`` ALONE are reported separately. If that number is not zero, it is the single
most important thing on the page.

    PYTHONPATH=. python tools/track_level_birds.py \
        --gt work/gt_user.json \
        --tracks work/tracks3/0705/pc-max-all.json \
        --dets work/det3/0705/pc-max.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dronedet import trackclass as TC  # noqa: E402
from dronedet.console import use_utf8_stdio  # noqa: E402
from dronedet.gt import GroundTruth  # noqa: E402

TAU = 16.0   # same association radius tools/eval_tracks.py uses, for comparability


def _radius(w: float, h: float) -> float:
    return max(TAU, 0.5 * math.sqrt(max(w * h, 1.0)))


def classify_gt(gt: GroundTruth, bird_prefix: str) -> dict[str, str]:
    """-> {object_name: 'target' | 'bird' | 'other-ignore'}.

    NOT every ``ignore`` object is a bird, and conflating them produces a report that is
    exactly backwards. 07_05's ignore set is eight `bird*` tracks (934 instances) **plus
    `near`** -- the *landed drone*, 571 instances, flagged ignore because it is the same
    aircraft sitting on the ground rather than a distractor. Treating `near` as a bird
    made this tool announce "1478 detections on birds" out of 1482 bird instances and a
    bird track classified `near`, when what it had actually found was the detector
    correctly finding the landed drone and the classifier correctly labelling it.

    So birds are identified by name prefix, which is the convention
    `dronedet.metrics.Summary.confuser_hits(prefixes=("bird",))` already uses, and every
    other ignore object gets its own category rather than being silently counted as a
    false alarm.
    """
    out = {}
    for name, obj in gt.objects.items():
        if not obj.ignore:
            out[name] = "target"
        elif name.lower().startswith(bird_prefix):
            out[name] = "bird"
        else:
            out[name] = "other-ignore"
    return out


def gt_kind_of_track(tr: dict, gt: GroundTruth, excl: set[int],
                     kinds: dict[str, str]) -> tuple[str, str, dict]:
    """-> (kind, object_name, fractions). kind in target/bird/other-ignore/nothing/ambiguous.

    A target outranks a distractor the track also overlaps, which is the convention in
    `dronedet.metrics._match_frame`: a detection is never stolen by a distractor it
    happens to sit near.
    """
    counts = {"target": 0, "bird": 0, "other-ignore": 0}
    total = 0
    hits: dict[str, int] = {}
    for f, v in tr["frames"].items():
        fi = int(f)
        if fi in excl:
            continue
        total += 1
        hit = None
        for name, obj in gt.objects.items():
            b = obj.box(fi)
            if b is None:
                continue
            if math.hypot(v[0] - b[0], v[1] - b[1]) <= _radius(b[2], b[3]):
                k = kinds[name]
                if k == "target":
                    hit = (k, name)
                    break
                hit = hit or (k, name)
        if hit:
            hits[hit[1]] = hits.get(hit[1], 0) + 1
            counts[hit[0]] += 1
    if not total:
        return "nothing", "", {"target": 0.0, "bird": 0.0, "other-ignore": 0.0,
                               "nothing": 1.0}
    fr = {k: v / total for k, v in counts.items()}
    fr["nothing"] = 1.0 - sum(fr.values())
    kind = next((k for k, v in fr.items() if v > 0.5), "ambiguous")
    who = max(hits, key=hits.get) if hits else ""
    return kind, who, {k: round(v, 3) for k, v in fr.items()}


def raw_bird_detections(gt: GroundTruth, dets: dict, excl: set[int], conf: float,
                        kinds: dict[str, str]) -> dict:
    """Per-frame detections landing on each GT kind, BEFORE any tracking.

    The baseline the track-level number is measured against: how much bird evidence the
    detector produced in the first place. A track-level bird count of zero means nothing
    if the detector never fired on a bird at all.
    """
    tally = {"target": 0, "bird": 0, "other-ignore": 0}
    total = 0
    per_bird: dict[str, int] = {}
    for f, ds in dets["frames"].items():
        fi = int(f)
        if fi in excl:
            continue
        for d in ds:
            if d[4] < conf:
                continue
            total += 1
            best = None
            cx, cy = (d[0] + d[2]) / 2, (d[1] + d[3]) / 2
            for name, obj in gt.objects.items():
                b = obj.box(fi)
                if b is None:
                    continue
                if math.hypot(cx - b[0], cy - b[1]) <= _radius(b[2], b[3]):
                    k = kinds[name]
                    if k == "target":
                        best = (k, name)
                        break
                    best = best or (k, name)
            if best:
                tally[best[0]] += 1
                if best[0] == "bird":
                    per_bird[best[1]] = per_bird.get(best[1], 0) + 1
    return {"detections_scored": total, **{f"on_{k}": v for k, v in tally.items()},
            "on_nothing": total - sum(tally.values()), "per_bird": per_bird}


def main() -> int:
    use_utf8_stdio()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True, type=Path)
    ap.add_argument("--tracks", required=True, type=Path)
    ap.add_argument("--dets", required=True, type=Path)
    ap.add_argument("--conf", type=float, default=0.25,
                    help="detection floor for the RAW bird count (the tracker applies "
                         "its own min_score independently)")
    ap.add_argument("--bird-prefix", default="bird",
                    help="GT objects whose name starts with this AND are ignore-flagged "
                         "are distractor birds. Other ignore objects (e.g. `near`, the "
                         "landed drone) are neither targets nor false alarms.")
    ap.add_argument("--no-motion", action="store_true",
                    help="disable the motion clause in the classifier (see "
                         "trackclass.classify_tracks allow_motion)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    gt = GroundTruth.load(a.gt)
    tracks = json.loads(a.tracks.read_text(encoding="utf-8"))
    dets = json.loads(a.dets.read_text(encoding="utf-8"))
    excl = set(gt.meta.get("exclude_frames", []))

    kinds = classify_gt(gt, a.bird_prefix)
    birds = {n for n, k in kinds.items() if k == "bird"}
    targets = {n for n, k in kinds.items() if k == "target"}
    other_ign = {n for n, k in kinds.items() if k == "other-ignore"}
    bird_instances = sum(len([f for f in gt.objects[n].frames if f not in excl])
                         for n in birds)

    cls = TC.classify_tracks(tracks, dets, allow_motion=not a.no_motion)

    rows = []
    for tr in tracks["tracks"]:
        c = cls.get(tr["id"], {})
        kind, who, fr = gt_kind_of_track(tr, gt, excl, kinds)
        # Re-derive WHICH clause promoted this track. classify_tracks returns only the
        # verdict, and the whole point here is to attribute it.
        confirmed = (c.get("conf_frac", 0) >= TC.CONF_FRAC
                     and c.get("n_conf", 0) >= TC.N_CONF)
        sustained = c.get("n_tracked", 0) >= TC.LONG_TRACK
        rows.append({
            "id": tr["id"], "pred": c.get("cls", "?"), "gt_kind": kind,
            "gt_object": who, "fractions": fr,
            "n_frames": c.get("n", len(tr["frames"])),
            "n_tracked": c.get("n_tracked", 0),
            "conf_frac": c.get("conf_frac", 0.0), "n_conf": c.get("n_conf", 0),
            "confirmed_by_appearance": confirmed,
            "sustained_only": sustained and not confirmed,
        })

    # ---------------------------------------------------------------- confusion matrix
    preds = ["drone", "near", "other"]
    gt_kinds = ["target", "bird", "other-ignore", "nothing", "ambiguous"]
    cm = {p: {k: 0 for k in gt_kinds} for p in preds}
    for r in rows:
        if r["pred"] in cm and r["gt_kind"] in cm[r["pred"]]:
            cm[r["pred"]][r["gt_kind"]] += 1

    raised = [r for r in rows if r["pred"] in ("drone", "near")]
    tp = [r for r in raised if r["gt_kind"] == "target"]
    fp_bird = [r for r in raised if r["gt_kind"] == "bird"]
    fp_other = [r for r in raised if r["gt_kind"] in ("nothing", "ambiguous")]
    # A raised track on the LANDED drone is not a false alarm -- it is the same aircraft,
    # flagged ignore because it is on the ground. It is excluded from both precision
    # numerator and denominator rather than scored either way, which is what `ignore`
    # means everywhere else in this repo.
    on_other_ign = [r for r in raised if r["gt_kind"] == "other-ignore"]
    judged = [r for r in raised if r["gt_kind"] != "other-ignore"]
    # Recall at the track level: a target object is recalled if SOME raised track sits
    # on it. Counting raised-tracks-per-target instead would let one object rescue
    # another, and would reward id switches.
    covered = {r["gt_object"] for r in tp}
    missed = [n for n in targets if n not in covered]

    def dur(rs):
        v = [r["n_frames"] for r in rs]
        return {"n": len(v), "median": st.median(v) if v else None,
                "min": min(v) if v else None, "max": max(v) if v else None}

    prec = len(tp) / max(len(judged), 1)
    rec = len(covered) / max(len(targets), 1)
    report = {
        "gt": str(a.gt), "tracks": str(a.tracks), "dets": str(a.dets),
        "constants": {"CONF_FRAC": TC.CONF_FRAC, "N_CONF": TC.N_CONF,
                      "LONG_TRACK": TC.LONG_TRACK, "DRONE_SCORE": TC.DRONE_SCORE,
                      "allow_motion": not a.no_motion},
        "ground_truth": {"target_objects": sorted(targets),
                         "bird_objects": sorted(birds),
                         "other_ignore_objects": sorted(other_ign),
                         "bird_instances": bird_instances},
        "raw_detections": raw_bird_detections(gt, dets, excl, a.conf, kinds),
        "n_confirmed_tracks": len(tracks["tracks"]),
        "confusion_matrix": cm,
        "operating_point": {
            "raised_tracks": len(raised),
            "raised_on_other_ignore_not_scored": len(on_other_ign),
            "judged_tracks": len(judged),
            "true_target_tracks": len(tp),
            "bird_false_alarms": len(fp_bird),
            "clutter_false_alarms": len(fp_other),
            "track_precision": round(prec, 4),
            "track_recall_over_objects": round(rec, 4),
            "missed_target_objects": missed,
        },
        "durations_frames": {"target": dur(tp), "bird": dur(fp_bird),
                             "clutter": dur(fp_other)},
        "long_track_bypass": {
            "promoted_by_sustained_only": sum(1 for r in rows if r["sustained_only"]),
            "of_which_bird": sum(1 for r in rows
                                 if r["sustained_only"] and r["gt_kind"] == "bird"),
            "of_which_clutter": sum(1 for r in rows if r["sustained_only"]
                                    and r["gt_kind"] in ("nothing", "ambiguous")),
        },
        "rejected_tracks": {
            "n_other": sum(1 for r in rows if r["pred"] == "other"),
            # Why each rejection happened. Both clauses can fail at once, so these are
            # not mutually exclusive and are reported as counts, not a partition.
            "failed_conf_frac": sum(1 for r in rows if r["pred"] == "other"
                                    and r["conf_frac"] < TC.CONF_FRAC),
            "failed_n_conf": sum(1 for r in rows if r["pred"] == "other"
                                 and r["n_conf"] < TC.N_CONF),
            "rejected_that_were_birds": sum(1 for r in rows if r["pred"] == "other"
                                            and r["gt_kind"] == "bird"),
            "rejected_that_were_targets": sum(1 for r in rows if r["pred"] == "other"
                                              and r["gt_kind"] == "target"),
        },
        "tracks": rows,
    }

    L = ["# Track-level bird / false-positive analysis", "",
         f"`{a.tracks.name}` scored against `{a.gt.name}`. "
         f"CONF_FRAC={TC.CONF_FRAC}, N_CONF={TC.N_CONF}, LONG_TRACK={TC.LONG_TRACK}, "
         f"DRONE_SCORE={TC.DRONE_SCORE}.", "",
         "A bird raised for 150 consecutive frames is ONE false alarm to an operator, "
         "not 150. Every number below counts tracks.", "",
         f"Ground truth: {len(targets)} target object(s) {sorted(targets)}, "
         f"{len(birds)} labelled bird track(s) with {bird_instances} instances, and "
         f"{len(other_ign)} other ignore object(s) {sorted(other_ign)} — "
         "ignore-flagged but **not** distractors (the landed drone is the same aircraft "
         "on the ground), so they are scored neither for nor against.", "",
         "## Before tracking -- what the detector produced", "",
         f"| detections >= {a.conf} | on target | on bird | on other-ignore | on nothing |",
         "|---|---|---|---|---|"]
    rd = report["raw_detections"]
    L += [f"| {rd['detections_scored']} | {rd['on_target']} | {rd['on_bird']} | "
          f"{rd['on_other-ignore']} | {rd['on_nothing']} |", "",
          "## After tracking -- confusion matrix", "",
          "| classified as | " + " | ".join(gt_kinds) + " |",
          "|---|" + "---|" * len(gt_kinds)]
    for p in preds:
        L.append(f"| **{p}** | " + " | ".join(str(cm[p][k]) for k in gt_kinds) + " |")
    op = report["operating_point"]
    L += ["", "## Operating point", "",
          f"- tracks raised as a target: **{op['raised_tracks']}** "
          f"({op['raised_on_other_ignore_not_scored']} of them on an ignore object that "
          f"is not a distractor, excluded from the scoring below)",
          f"- judged: **{op['judged_tracks']}**",
          f"- of those, genuinely the drone: **{op['true_target_tracks']}**",
          f"- **bird false alarms: {op['bird_false_alarms']}**",
          f"- clutter false alarms: {op['clutter_false_alarms']}",
          f"- track precision: **{op['track_precision']:.3f}**",
          f"- target objects recovered: **{op['track_recall_over_objects']:.3f}** "
          f"({len(covered)}/{len(targets)})", ""]
    lb = report["long_track_bypass"]
    L += ["## The LONG_TRACK bypass", "",
          f"Tracks promoted with no appearance evidence at all, purely for lasting "
          f">= {TC.LONG_TRACK} frames: **{lb['promoted_by_sustained_only']}** "
          f"(birds: {lb['of_which_bird']}, clutter: {lb['of_which_clutter']}).", ""]
    if lb["of_which_bird"]:
        L += ["**That clause raised a labelled bird as a target.** It is the one rule in "
              "the classifier with no appearance requirement, and this is what it costs.",
              ""]

    out = "\n".join(L) + "\n"
    print(out)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(out, encoding="utf-8")
        a.out.with_suffix(".json").write_text(json.dumps(report, indent=2),
                                              encoding="utf-8")
        print(f"wrote {a.out} and {a.out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
