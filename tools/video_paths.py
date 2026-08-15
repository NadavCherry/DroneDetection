"""Resolving a ground-truth sequence name to the video file it refers to. One resolver.

WHY THIS IS ITS OWN MODULE
--------------------------
Both arms of the benchmark have to turn a GT stem into a video path, and they run in
DIFFERENT python environments: ours in `speck` (torch 2.x, ultralytics), the competitor's
in `tph` (YOLOv5-era, numpy < 2). `tools/infer_tiled` imports `dronedet` and cannot be
imported from the competitor's environment, so the resolver used to be duplicated -- once
properly, once as a five-extension bash loop in an sbatch.

That asymmetry is not cosmetic. Our arm resolved a video and scored it; the competitor's
loop missed the same file, dropped it, and `tools/evaluate.py` scored the whole sequence
as a TOTAL MISS with its full ground truth still charged to the denominator. Every dropped
clip deflates the COMPETITOR's AP only -- the direction that ends in a retraction rather
than a missed opportunity.

So: no heavy imports here, nothing but pathlib, and both arms call the same function.

THE NAMING TRAP
---------------
NPS ships `Clip_41.mov` while Dogfight's annotations -- and therefore our GT files -- say
`Clip_041`. A detection JSON must be named after the GT STEM, not the video stem, because
`tools/evaluate.py` pairs them by filename. Resolving the video correctly and then naming
the output `Clip_41.json` is a fix that looks applied and still scores every sequence as a
total miss. `resolve_all` returns (gt_stem, path) pairs so callers cannot get this wrong.
"""

from __future__ import annotations

from pathlib import Path

#: Containers seen across the corpora this repo scores. ARD-MAV ships .mp4, NPS ships .mov,
#: and case varies between releases -- the cluster is case-sensitive, so both are listed.
VIDEO_EXTS = (".mp4", ".mov", ".MOV", ".MP4", ".avi", ".AVI", ".m4v", ".mkv")


def resolve_video(root: Path, stem: str) -> Path | None:
    """The video for a GT stem, whatever container, case, or zero-padding it uses."""
    root = Path(root)
    for ext in VIDEO_EXTS:
        p = root / f"{stem}{ext}"
        if p.exists():
            return p

    # NPS: Dogfight's annotations say Clip_041; Purdue's file on disk is Clip_41.mov.
    if "_" in stem:
        head, _, tail = stem.rpartition("_")
        if tail.isdigit():
            for cand in (f"{head}_{int(tail)}", f"{head}_{int(tail):03d}"):
                if cand == stem:
                    continue
                for ext in VIDEO_EXTS:
                    p = root / f"{cand}{ext}"
                    if p.exists():
                        return p

    # Last resort: an exact-stem glob. Deliberately `f"{stem}.*"` and not `f"{stem}*"` --
    # a prefix match would let Clip_04 resolve to Clip_041.mov and score one sequence's
    # detections against another sequence's ground truth.
    hits = sorted(p for p in root.glob(f"{stem}.*") if p.is_file())
    return hits[0] if hits else None


def resolve_all(root: Path, gt_dir: Path) -> tuple[list[tuple[str, Path]], list[str]]:
    """Every GT stem in `gt_dir` paired with its video. -> (pairs, unresolved_stems).

    Pairs carry the GT stem, not the video stem, so a caller naming its output after
    `pair[0]` stays matched to the ground truth it will be scored against.
    """
    pairs, missing = [], []
    for gt in sorted(Path(gt_dir).glob("*.json")):
        v = resolve_video(root, gt.stem)
        (pairs.append((gt.stem, v)) if v is not None else missing.append(gt.stem))
    return pairs, missing


__all__ = ["VIDEO_EXTS", "resolve_all", "resolve_video"]
