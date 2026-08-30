"""The scorecard: one model, one benchmark, stored so it can still be argued with later.

The shape of this file is the whole point. A scorecard that stored only summary numbers
would make every downstream question unanswerable without re-running the model:

* **Per-sequence, not pooled.** The resampling unit for a paired comparison is the
  *sequence*, because frames within one flight are strongly correlated. A scorecard that
  pooled everything into one AP could never support a confidence interval that means
  anything -- which is exactly how this project ended up reporting AP 1.000 with no
  spread from what was really a single flight.
* **Per-detection outcome, not per-threshold summaries.** Every detection is stored as
  ``(score, outcome)``. From that, AP, precision/recall at *any* threshold, and the bird
  false-alarm count at *any* threshold are all recomputable. Storing "bird hits at 0.55"
  instead would freeze a choice that later turns out to be the wrong operating point.
* **Conditions travel with the sequence.** "Works at night" is a claim about a subset of
  sequences, so the subset membership has to be in the artifact, not reconstructed from
  memory later.

Outcomes are ``'tp'``, ``'fp'``, or ``'distractor:<object-name>'``. Distractors are real
things the detector must not call a drone -- a bird, a plane, the near/landed drone. They
never count toward recall, and hits on them are counted and reported rather than
discarded (`dronedet/evaluate.py` used to drop them silently, which hid the single
result this project most wants to claim).

Stdlib only, so it loads in the torch-free CI job.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCHEMA_VERSION = 1


@dataclass
class SequenceResult:
    """One model's behaviour on one sequence, stored losslessly enough to re-argue."""
    sequence: str
    n_gt: int
    n_frames: int
    conditions: list[str] = field(default_factory=list)
    detections: list[tuple[float, str]] = field(default_factory=list)
    distractor_instances: dict[str, int] = field(default_factory=dict)
    target_px_median: float | None = None

    def scored(self) -> list[tuple[float, str]]:
        """Detections that count toward precision -- tp and fp only."""
        return [(s, o) for s, o in self.detections if o in ("tp", "fp")]

    def distractor_hits(self, threshold: float = 0.0,
                        prefixes: tuple[str, ...] = ()) -> int:
        """Hits on distractor objects at a threshold, optionally filtered by name prefix.

        ``prefixes=('bird',)`` gives the bird false-alarm count. Empty means all
        distractors, which lumps benign ones (the near drone) in with confusers -- almost
        never what you want in a headline.
        """
        n = 0
        for s, o in self.detections:
            if s < threshold or not o.startswith("distractor:"):
                continue
            name = o.split(":", 1)[1]
            if not prefixes or name.startswith(prefixes):
                n += 1
        return n

    def distractor_total(self, prefixes: tuple[str, ...] = ()) -> int:
        return sum(v for k, v in self.distractor_instances.items()
                   if not prefixes or k.startswith(prefixes))


@dataclass
class Scorecard:
    """Everything one evaluation run produced, plus what makes it interpretable."""
    model: str
    dataset_key: str
    protocol_key: str
    split: str
    sequences: list[SequenceResult] = field(default_factory=list)
    # provenance -- without these a scorecard is an assertion, not a measurement
    git_sha: str = ""
    git_dirty: bool = False
    weights_sha256: str = ""
    command: str = ""
    created: str = ""
    seed: int | None = None
    notes: str = ""
    schema_version: int = SCHEMA_VERSION

    # ------------------------------------------------------------------ selection
    def with_condition(self, condition: str) -> "Scorecard":
        """A view restricted to sequences carrying a condition, e.g. 'night'.

        This is how a condition-stratified table is built: the same statistics, over a
        subset chosen by a property of the data rather than of the result.
        """
        return Scorecard(
            model=self.model, dataset_key=self.dataset_key, protocol_key=self.protocol_key,
            split=f"{self.split}[{condition}]",
            sequences=[s for s in self.sequences if condition in s.conditions],
            git_sha=self.git_sha, git_dirty=self.git_dirty, weights_sha256=self.weights_sha256,
            command=self.command, created=self.created, seed=self.seed,
            notes=self.notes, schema_version=self.schema_version)

    def conditions_present(self) -> list[str]:
        seen: set[str] = set()
        for s in self.sequences:
            seen.update(s.conditions)
        return sorted(seen)

    # ------------------------------------------------------------------ aggregates
    @property
    def n_gt(self) -> int:
        return sum(s.n_gt for s in self.sequences)

    @property
    def n_frames(self) -> int:
        return sum(s.n_frames for s in self.sequences)

    @property
    def n_sequences(self) -> int:
        return len(self.sequences)

    def pooled_detections(self) -> list[tuple[float, str]]:
        out: list[tuple[float, str]] = []
        for s in self.sequences:
            out.extend(s.scored())
        return out

    def distractor_hits(self, threshold: float = 0.0,
                        prefixes: tuple[str, ...] = ()) -> tuple[int, int]:
        """(hits, instances on offer) across the whole scorecard."""
        hits = sum(s.distractor_hits(threshold, prefixes) for s in self.sequences)
        total = sum(s.distractor_total(prefixes) for s in self.sequences)
        return hits, total

    # ------------------------------------------------------------------ io
    #
    # Scorecards are stored gzipped when the path says so, and `load` accepts either form.
    #
    # This is not premature optimisation. The artifact holds every detection down to the
    # 0.001 confidence floor, which is the property that makes precision at any operating
    # point recomputable -- and also means two thirds of every file is a tail of near-zero
    # false positives. The campaign's 42 scorecards are 623.7 MB raw and 29.1 MB gzipped,
    # a 21.5x saving on text that is almost entirely repeated digits. Raw, they cannot ship
    # in a repository that is already 510 MB, and .gitignore's own rule says "a scorecard
    # you intend to CITE should be force-added, because a number in a report must point at
    # the artifact that produced it". Gzip is what makes that rule affordable rather than
    # aspirational; nothing is lost, the bytes are identical after decompression.
    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        blob = json.dumps(asdict(self), indent=1)
        if p.suffix == ".gz":
            import gzip
            p.write_bytes(gzip.compress(blob.encode("utf-8"), 6))
        else:
            p.write_text(blob, encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "Scorecard":
        p = Path(path)
        # Accept either form, and fall back to the gzipped sibling so that every existing
        # caller naming a plain .json keeps working once the artifact is compressed.
        if not p.exists() and p.suffix != ".gz":
            gz = p.with_suffix(p.suffix + ".gz")
            if gz.exists():
                p = gz
        if p.suffix == ".gz":
            import gzip
            d = json.loads(gzip.decompress(p.read_bytes()).decode("utf-8"))
        else:
            d = json.loads(p.read_text(encoding="utf-8"))
        got = d.get("schema_version", 0)
        if got != SCHEMA_VERSION:
            raise ValueError(
                f"{p}: scorecard schema v{got}, this code expects v{SCHEMA_VERSION}. "
                "Re-run the evaluation rather than reading it with mismatched semantics.")
        seqs = [SequenceResult(**{**s, "detections": [tuple(x) for x in s["detections"]]})
                for s in d.pop("sequences")]
        return Scorecard(sequences=seqs, **d)


# ---------------------------------------------------------------------- statistics


def average_precision(detections: list[tuple[float, str]], n_gt: int) -> float:
    """VOC all-point AP from ``(score, outcome)`` pairs. Distractors are already excluded.

    Duplicated deliberately from `dronedet.metrics` rather than imported: that one works
    on `Record` objects tied to a live evaluation, this one works on the compact stored
    form, and coupling them would force the scorecard to carry the heavier type.
    """
    scored = [(s, o) for s, o in detections if o in ("tp", "fp")]
    if not scored or n_gt <= 0:
        return 0.0
    scored.sort(key=lambda r: -r[0])
    tp = fp = 0
    prev_recall = 0.0
    points: list[tuple[float, float]] = []
    for s, o in scored:
        tp += o == "tp"
        fp += o == "fp"
        points.append((tp / n_gt, tp / max(tp + fp, 1)))
    # monotone precision envelope, then integrate over recall
    best = 0.0
    enveloped = []
    for recall, precision in reversed(points):
        best = max(best, precision)
        enveloped.append((recall, best))
    enveloped.reverse()
    ap = 0.0
    for recall, precision in enveloped:
        ap += (recall - prev_recall) * precision
        prev_recall = recall
    return float(ap)


def pooled_ap(sequences: list[SequenceResult]) -> float:
    """AP over a collection of sequences, pooling their detections.

    This is the statistic to hand to `dronedet.stats.paired_bootstrap_diff`: it takes the
    *resampled collection*, so pooled metrics stay pooled instead of becoming a mean of
    per-sequence APs (which weights a 30-frame sequence like a 3,000-frame one).
    """
    dets: list[tuple[float, str]] = []
    n_gt = 0
    for s in sequences:
        dets.extend(s.scored())
        n_gt += s.n_gt
    return average_precision(dets, n_gt)


def pooled_recall(sequences: list[SequenceResult], threshold: float) -> float:
    tp = sum(1 for s in sequences for sc, o in s.detections if o == "tp" and sc >= threshold)
    n_gt = sum(s.n_gt for s in sequences)
    return tp / n_gt if n_gt else 0.0


def pooled_precision(sequences: list[SequenceResult], threshold: float) -> float:
    tp = fp = 0
    for s in sequences:
        for sc, o in s.detections:
            if sc < threshold:
                continue
            tp += o == "tp"
            fp += o == "fp"
    return tp / (tp + fp) if (tp + fp) else 0.0
