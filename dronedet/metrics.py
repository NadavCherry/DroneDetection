"""Benchmark-grade detection metrics for few-pixel targets.

`evaluate.py` answers one question well -- "does this method find *our* drone" --
with one matching rule (centre distance) and one summary (AP). It is not enough to
compare against published work, for three reasons this module fixes:

1. **Published numbers are IoU-based.** Every drone paper reports AP@0.5 IoU (or COCO
   AP@[.5:.95]). Centre-distance AP is the *right* metric at 4 px and this repo keeps
   it, but a number that cannot be placed beside a published one cannot support a
   comparison. Both are computed here, from the same detections, side by side.

2. **Distractor hits are invisible.** `evaluate.py` drops detections that land on an
   ``ignore`` object -- neither TP nor FP (`evaluate.py:54`). On 07_05 that silently
   discards every hit on the 8 labelled bird tracks (934 boxes, median 6.0 px -- the
   same size band as the 8.0 px drone). Bird rejection is the hardest thing this
   pipeline does and it was going unscored. Here ``distractor`` is a third outcome:
   excluded from precision by default, but always *counted and reported*, so
   "0 hits on 934 bird instances" becomes a claim instead of a silence.

3. **No spread.** A single AP over one video reads as a benchmark result when it is a
   case study. `bootstrap_ci` resamples contiguous frame blocks (never single frames --
   consecutive frames of one track are not independent samples) so the interval
   reflects roughly how many *independent looks* at the target there were.

Size bins follow AI-TOD (Wang et al.), the standard for this scale, on sqrt(w*h):
very-tiny 2-8 px, tiny 8-16, small 16-32, medium 32+. 51 % of the 07_05 drone
instances fall in the very-tiny bin, so an unbinned AP hides where a method fails.

Conventions
-----------
* **Ground-truth boxes are ``(cx, cy, w, h)``** -- centre, not corner (`dronedet.gt`
  stores them that way; `dronedet.detections.Detection` is corner-based ``xyxy``).
  Everything here is converted to ``xyxy`` on the way in, because mixing the two
  conventions costs ~w/2 px of phantom localisation error, which at a 6 px target is
  the whole measurement.
* ``ignore`` objects are *distractors*: real things the detector must not call a
  drone. They never contribute to recall.
* An operating threshold must come from `pick_threshold` on a **validation** set and
  be passed to `summarise` for the test set. Sweeping the threshold on the set you
  report is how `evaluate.report` chooses its best-F1 point, which makes that
  precision/recall pair an oracle rather than an achievable operating point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# AI-TOD size bins, on sqrt(area) in pixels.
SIZE_BINS: tuple[tuple[str, float, float], ...] = (
    ("very-tiny", 2.0, 8.0),
    ("tiny", 8.0, 16.0),
    ("small", 16.0, 32.0),
    ("medium", 32.0, float("inf")),
)

COCO_IOU_THRESHOLDS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))


def size_bin(w: float, h: float) -> str:
    s = math.sqrt(max(w * h, 0.0))
    for name, lo, hi in SIZE_BINS:
        if lo <= s < hi:
            return name
    return "very-tiny" if s < 2.0 else "medium"


def cxcywh_to_xyxy(b: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """GT convention ``(cx, cy, w, h)`` -> ``(x1, y1, x2, y2)``."""
    cx, cy, w, h = b
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def iou(a: tuple[float, float, float, float],
        b: tuple[float, float, float, float]) -> float:
    """IoU of two ``xyxy`` boxes."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = max(ax1 - ax0, 0) * max(ay1 - ay0, 0) + \
        max(bx1 - bx0, 0) * max(by1 - by0, 0) - inter
    return inter / union if union > 0 else 0.0


def centre_distance_ok(det: tuple[float, float, float, float],
                       gt: tuple[float, float, float, float],
                       tau: float) -> tuple[bool, float]:
    """Match if the detection centre lies within ``max(tau, 0.5*sqrt(gt area))``.

    Both boxes are ``xyxy``. Returns ``(matched, centre_error_px)``. This is the
    repo's native rule: a 1 px shift on a 6 px box swings IoU wildly, so localisation
    is scored by how far the centre is, not by box overlap.
    """
    dx0, dy0, dx1, dy1 = det
    gx0, gy0, gx1, gy1 = gt
    err = math.hypot((dx0 + dx1) / 2 - (gx0 + gx1) / 2,
                     (dy0 + dy1) / 2 - (gy0 + gy1) / 2)
    radius = max(tau, 0.5 * math.sqrt(max((gx1 - gx0) * (gy1 - gy0), 1.0)))
    return err <= radius, err


@dataclass
class Record:
    """One detection, resolved against the ground truth of its frame."""
    frame: int
    score: float
    outcome: str            # 'tp' | 'fp' | 'distractor'
    obj: str | None = None  # GT object name for tp/distractor
    error: float = float("nan")   # centre error px (tp only)
    bin: str = ""           # size bin of the matched GT (tp only)
    #: sqrt(area) of the matched GT box, in pixels; NaN for a false positive, which has
    #: no size. Kept alongside ``bin`` so a caller can re-bin at edges other than the
    #: AI-TOD ones without re-running the match -- the accuracy-vs-target-size curve
    #: needs finer resolution around 8-25 px than the four standard bins provide.
    gt_size: float = float("nan")


@dataclass
class Evaluation:
    records: list[Record]
    n_gt: int
    n_frames: int
    gt_per_bin: dict[str, int] = field(default_factory=dict)
    gt_per_obj: dict[str, int] = field(default_factory=dict)
    gt_per_frame: dict[int, int] = field(default_factory=dict)
    distractor_instances: int = 0   # how many distractor boxes were on offer
    distractor_instances_by_object: dict[str, int] = field(default_factory=dict)
    frames: list[int] = field(default_factory=list)
    #: sqrt(area) in px of every TARGET GT instance, in encounter order. ``gt_per_bin`` is
    #: this list histogrammed at the AI-TOD edges; keeping the raw values lets a caller
    #: re-bin at any edges and still get an exact denominator, which a histogram cannot.
    gt_sizes: list[float] = field(default_factory=list)


def _match_frame(dets, gts, rule, tau, iou_thr):
    """Greedy score-ordered assignment of detections to GT boxes in one frame.

    ``dets``: list of ``(x1, y1, x2, y2, score)``.
    ``gts``:  dict ``name -> (x1, y1, x2, y2, is_distractor)``.
    Positives outrank distractors at equal match quality, so a detection is never
    stolen by a distractor it also overlaps; each GT object may be claimed once.
    """
    order = sorted(range(len(dets)), key=lambda i: -dets[i][4])
    used: set[str] = set()
    out: dict[int, tuple] = {}
    for i in order:
        det_box = dets[i][:4]
        best, best_key, best_err = None, None, float("nan")
        for name, gt in gts.items():
            if name in used:
                continue
            gt_box, distract = gt[:4], gt[4]
            if rule == "centre":
                ok, err = centre_distance_ok(det_box, gt_box, tau)
                key = -err          # nearer is better
            else:
                v = iou(det_box, gt_box)
                ok, err, key = v >= iou_thr, v, v
            if not ok:
                continue
            # positives outrank distractors at equal quality
            rank = (0 if distract else 1, key)
            if best is None or rank > best:
                best, best_key, best_err = rank, name, err
        if best_key is None:
            out[i] = ("fp", None, float("nan"))
        elif gts[best_key][4]:
            out[i] = ("distractor", best_key, best_err)
        else:
            used.add(best_key)
            out[i] = ("tp", best_key, best_err)
    return out


def evaluate(gt, dets, *, rule: str = "centre", tau: float = 12.0,
             iou_thr: float = 0.5, targets: set[str] | None = None,
             frame_range: tuple[int, int] | None = None) -> Evaluation:
    """Resolve every detection into tp / fp / distractor.

    ``gt``   : ``dronedet.gt.GroundTruth`` (or anything with ``.objects`` and ``.meta``).
    ``dets`` : ``dronedet.detections.DetectionSet``.
    ``rule`` : ``'centre'`` (this repo's native rule) or ``'iou'`` (comparable to papers).
    ``targets``: object names that count as positives; every other object becomes a
      distractor. Defaults to every object not flagged ``ignore`` in the GT.
    """
    if rule not in ("centre", "iou"):
        raise ValueError(f"rule must be 'centre' or 'iou', got {rule!r}")
    excluded = set(gt.meta.get("exclude_frames", []))
    frames = sorted(set(dets.frames) - excluded)
    if frame_range is not None:
        lo, hi = frame_range
        frames = [f for f in frames if lo <= f < hi]

    records: list[Record] = []
    n_gt = 0
    n_distract = 0
    per_bin: dict[str, int] = {}
    per_obj: dict[str, int] = {}
    per_frame: dict[int, int] = {}
    distract_per_obj: dict[str, int] = {}
    gt_sizes: list[float] = []

    for f in frames:
        gts = {}
        per_frame[f] = 0
        for name, obj in gt.objects.items():
            box = obj.box(f)
            if box is None:
                continue
            is_target = (name in targets) if targets is not None else (not obj.ignore)
            gts[name] = (*cxcywh_to_xyxy(box), not is_target)   # GT is (cx,cy,w,h)
            if is_target:
                n_gt += 1
                per_frame[f] += 1
                b = size_bin(box[2], box[3])
                per_bin[b] = per_bin.get(b, 0) + 1
                per_obj[name] = per_obj.get(name, 0) + 1
                gt_sizes.append(math.sqrt(max(box[2] * box[3], 0.0)))
            else:
                n_distract += 1
                distract_per_obj[name] = distract_per_obj.get(name, 0) + 1
        frame_dets = [(d.x1, d.y1, d.x2, d.y2, d.score) for d in dets.frames[f]]
        assigned = _match_frame(frame_dets, gts, rule, tau, iou_thr)
        for i, (outcome, name, err) in assigned.items():
            if name:
                gx0, gy0, gx1, gy1 = gts[name][:4]
                gt_bin = size_bin(gx1 - gx0, gy1 - gy0)
                gt_sz = math.sqrt(max((gx1 - gx0) * (gy1 - gy0), 0.0))
            else:
                gt_bin, gt_sz = "", float("nan")
            records.append(Record(frame=f, score=frame_dets[i][4], outcome=outcome,
                                  obj=name, error=err, bin=gt_bin, gt_size=gt_sz))

    return Evaluation(records=records, n_gt=n_gt, n_frames=len(frames),
                      gt_per_bin=per_bin, gt_per_obj=per_obj, gt_per_frame=per_frame,
                      distractor_instances=n_distract,
                      distractor_instances_by_object=distract_per_obj, frames=frames,
                      gt_sizes=gt_sizes)


def average_precision(records: list[Record], n_gt: int) -> float:
    """VOC all-point interpolated AP. Distractor records are dropped, not counted."""
    scored = [r for r in records if r.outcome in ("tp", "fp")]
    if not scored or n_gt == 0:
        return 0.0
    scored.sort(key=lambda r: -r.score)
    tp = np.cumsum([1 if r.outcome == "tp" else 0 for r in scored], dtype=float)
    fp = np.cumsum([1 if r.outcome == "fp" else 0 for r in scored], dtype=float)
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1e-12)
    # monotone precision envelope, then integrate over recall
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    prev_r = 0.0
    ap = 0.0
    for r, p in zip(recall, precision):
        ap += (r - prev_r) * p
        prev_r = r
    return float(ap)


def average_precision_11pt(records: list[Record], n_gt: int) -> float:
    """VOC2007 11-point interpolated AP: mean of max precision at recall >= t, for
    t in {0.0, 0.1, ..., 1.0}.

    Exists because GLAD -- the bar on ARD-MAV's official split -- defines AP this way, in
    its own words: "The AP is calculated at 0.5 IOU threshold and is averaged over
    uniformly spaced 11 points of the precision-recall curve." `average_precision` above
    is the all-point integral, a DIFFERENT quantity, and subtracting one from the other is
    exactly the mismatch `Protocol.mismatches_with` exists to refuse.

    The two disagree most where the curve is truncated: 11-point credits nothing above the
    highest recall reached, so a detector that stops at recall 0.75 scores zero at the
    three grid points above it. On a benchmark where this project's failure mode IS lost
    recall, that is not a rounding difference.
    """
    scored = [r for r in records if r.outcome in ("tp", "fp")]
    if not scored or n_gt == 0:
        return 0.0
    scored.sort(key=lambda r: -r.score)
    tp = np.cumsum([1 if r.outcome == "tp" else 0 for r in scored], dtype=float)
    fp = np.cumsum([1 if r.outcome == "fp" else 0 for r in scored], dtype=float)
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1e-12)
    total = 0.0
    for t in np.linspace(0.0, 1.0, 11):
        at = precision[recall >= t]
        total += float(at.max()) if at.size else 0.0
    return total / 11.0


def coco_ap(gt, dets, *, targets: set[str] | None = None,
            frame_range: tuple[int, int] | None = None) -> dict[str, float]:
    """AP averaged over IoU 0.50:0.05:0.95, plus AP50 and AP75.

    This is the number published drone papers report. On a 4 px target it is
    brutally pessimistic -- which is the point: it is what makes our result and
    theirs the same kind of number.
    """
    out: dict[str, float] = {}
    per_thr = []
    for thr in COCO_IOU_THRESHOLDS:
        ev = evaluate(gt, dets, rule="iou", iou_thr=thr, targets=targets,
                      frame_range=frame_range)
        ap = average_precision(ev.records, ev.n_gt)
        per_thr.append(ap)
        if thr == 0.5:
            out["AP50"] = ap
        elif thr == 0.75:
            out["AP75"] = ap
    out["AP"] = float(np.mean(per_thr))
    return out


def ap_by_size(ev: Evaluation) -> dict[str, float]:
    """AP restricted to each AI-TOD size bin.

    A false positive has no size, so it is charged to every bin: a method cannot
    look good on very-tiny targets by flooding the frame with large spurious boxes.
    """
    out: dict[str, float] = {}
    fps = [r for r in ev.records if r.outcome == "fp"]
    for name, _, _ in SIZE_BINS:
        n = ev.gt_per_bin.get(name, 0)
        if n == 0:
            continue
        subset = [r for r in ev.records if r.outcome == "tp" and r.bin == name] + fps
        out[name] = average_precision(subset, n)
    return out


#: Finer edges than AI-TOD's four bins, chosen for the accuracy-vs-target-size curve.
#: AI-TOD lumps 8-16 px into one "tiny" bin, but that interval is exactly where a
#: single-frame detector stops working and the interesting part of the curve lives, so
#: reporting it as one number hides the transition the curve exists to show.
MISSION_BINS: tuple[tuple[str, float, float], ...] = (
    ("<8 px", 0.0, 8.0),
    ("8-10 px", 8.0, 10.0),
    ("10-16 px", 10.0, 16.0),
    ("16-25 px", 16.0, 25.0),
    (">25 px", 25.0, float("inf")),
)


def ap_by_bins(ev: Evaluation, bins=SIZE_BINS) -> dict[str, tuple[float, int]]:
    """AP restricted to each of ``bins``, as ``{name: (ap, n_gt_in_bin)}``.

    Generalises `ap_by_size` to arbitrary edges, and returns the per-bin GT count with
    the AP because a bin's AP is uninterpretable without its denominator -- an AP of
    0.31 over 12 instances is noise, and reporting it beside an AP over 3,000 without
    saying so invites exactly the wrong conclusion.

    Two conventions worth stating, both inherited from `ap_by_size`:

    * A false positive has no size, so it is charged to EVERY bin. Otherwise a method
      could look strong on very-tiny targets by flooding the frame with large spurious
      boxes that no small-target bin ever pays for.
    * A true positive on an out-of-bin target is dropped rather than counted as a false
      positive -- it is neither a success nor a failure at this size. Counting it as a
      false positive (the naive way to subset) would depress every bin's AP by an amount
      that depends on how the OTHER bins are populated.
    """
    out: dict[str, tuple[float, int]] = {}
    fps = [r for r in ev.records if r.outcome == "fp"]
    for name, lo, hi in bins:
        n = sum(1 for s in ev.gt_sizes if lo <= s < hi)
        if n == 0:
            continue
        subset = [r for r in ev.records
                  if r.outcome == "tp" and lo <= r.gt_size < hi] + fps
        out[name] = (average_precision(subset, n), n)
    return out


def pick_threshold(ev: Evaluation) -> float:
    """The score threshold maximising F1 **on this (validation) evaluation**.

    Call this on val and pass the result to `summarise` for test. Choosing it on the
    set you report turns precision/recall into an oracle operating point.
    """
    scored = sorted((r for r in ev.records if r.outcome in ("tp", "fp")),
                    key=lambda r: -r.score)
    if not scored or ev.n_gt == 0:
        return 0.0
    best_f1, best_thr = -1.0, 0.0
    tp = fp = 0
    for i, r in enumerate(scored):
        tp += r.outcome == "tp"
        fp += r.outcome == "fp"
        if i + 1 < len(scored) and scored[i + 1].score == r.score:
            continue        # only threshold between score groups
        p = tp / max(tp + fp, 1)
        rec = tp / ev.n_gt
        f1 = 2 * p * rec / max(p + rec, 1e-12)
        if f1 > best_f1:
            best_f1, best_thr = f1, r.score
    return best_thr


@dataclass
class Summary:
    ap: float
    ap_by_size: dict[str, float]
    threshold: float
    precision: float
    recall: float
    f1: float
    fp_per_frame: float
    distractor_hits: int
    distractor_instances: int
    distractor_hits_per_frame: float
    median_centre_error: float
    n_gt: int
    n_frames: int
    recall_per_object: dict[str, float]
    # Per-object breakdown. Not every distractor is a confuser: on 07_05 the ``near``
    # object is the *same drone* landed and huge, so hitting it is correct behaviour,
    # while a hit on ``bird*`` is the failure the pipeline exists to prevent. Summing
    # them into one number hides exactly the result worth reporting.
    distractor_hits_by_object: dict[str, int] = field(default_factory=dict)
    distractor_instances_by_object: dict[str, int] = field(default_factory=dict)

    def confuser_hits(self, prefixes: tuple[str, ...] = ("bird",)) -> tuple[int, int]:
        """(hits, instances) restricted to distractors whose name starts with a prefix."""
        def keep(n):
            return any(n.startswith(p) for p in prefixes)
        hits = sum(v for n, v in self.distractor_hits_by_object.items() if keep(n))
        inst = sum(v for n, v in self.distractor_instances_by_object.items() if keep(n))
        return hits, inst


def summarise(ev: Evaluation, threshold: float) -> Summary:
    """Operating-point metrics at a threshold chosen elsewhere (see `pick_threshold`)."""
    kept = [r for r in ev.records if r.score >= threshold]
    tp = sum(r.outcome == "tp" for r in kept)
    fp = sum(r.outcome == "fp" for r in kept)
    dist = sum(r.outcome == "distractor" for r in kept)
    by_obj: dict[str, int] = {}
    for r in kept:
        if r.outcome == "distractor" and r.obj:
            by_obj[r.obj] = by_obj.get(r.obj, 0) + 1
    precision = tp / max(tp + fp, 1)
    recall = tp / max(ev.n_gt, 1)
    errs = [r.error for r in kept if r.outcome == "tp" and not math.isnan(r.error)]
    per_obj = {}
    for name, n in ev.gt_per_obj.items():
        per_obj[name] = sum(1 for r in kept if r.outcome == "tp" and r.obj == name) / max(n, 1)
    return Summary(
        ap=average_precision(ev.records, ev.n_gt),
        ap_by_size=ap_by_size(ev),
        threshold=threshold,
        precision=precision,
        recall=recall,
        f1=2 * precision * recall / max(precision + recall, 1e-12),
        fp_per_frame=fp / max(ev.n_frames, 1),
        distractor_hits=dist,
        distractor_instances=ev.distractor_instances,
        distractor_hits_per_frame=dist / max(ev.n_frames, 1),
        median_centre_error=float(np.median(errs)) if errs else float("nan"),
        n_gt=ev.n_gt,
        n_frames=ev.n_frames,
        recall_per_object=per_obj,
        distractor_hits_by_object=by_obj,
        distractor_instances_by_object=dict(ev.distractor_instances_by_object),
    )


def bootstrap_ci(ev: Evaluation, *, block: int = 30, n_resamples: int = 2000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile CI for AP from a moving-block bootstrap over frames.

    Consecutive frames of one flight are not independent -- a per-detection bootstrap
    would report a spuriously tight interval. Resampling contiguous blocks of
    ``block`` frames keeps the within-block correlation intact, so the interval
    reflects roughly ``n_frames / block`` independent looks at the target.

    This still cannot manufacture variety a single video does not contain: a tight
    interval here means "consistent within this flight", never "generalises".
    """
    if not ev.frames or ev.n_gt == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    by_frame: dict[int, list[Record]] = {}
    for r in ev.records:
        by_frame.setdefault(r.frame, []).append(r)

    frames = ev.frames
    starts = np.arange(max(len(frames) - block + 1, 1))
    n_blocks = max(len(frames) // block, 1)
    aps = []
    for _ in range(n_resamples):
        recs: list[Record] = []
        n_gt = 0
        for s in rng.choice(starts, size=n_blocks, replace=True):
            for f in frames[s:s + block]:
                recs.extend(by_frame.get(f, []))
                n_gt += ev.gt_per_frame.get(f, 0)
        aps.append(average_precision(recs, n_gt))
    lo = float(np.percentile(aps, 100 * alpha / 2))
    hi = float(np.percentile(aps, 100 * (1 - alpha / 2)))
    return lo, hi
