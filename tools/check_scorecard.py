#!/usr/bin/env python3
"""Refuse a scorecard that is a plumbing result rather than a model result.

Every eval job ends with this. It exists because the failure that has actually happened in
this project, twice, is not a crash -- it is a complete, well-formed scorecard reporting a
number that no model produced. A wrong `--video-root` gave six NPS scorecards reading
AP 0.000, because `tools/evaluate.py` correctly scores an unmatched sequence as a total
miss rather than skipping it. Nothing downstream could tell that from a genuinely bad
detector.

WHY NOT JUST READ scorecard["ap"]
---------------------------------
Because there is no such field, and the first version of this check read one. `Scorecard`
stores per-sequence `detections` and `n_gt`; AP is computed on demand by
`benchmarks.scorecard.pooled_ap`, so `d.get("ap")` was always None and the guard would
have aborted every job regardless of its result -- a guard that fires on success teaches
people to delete guards. It now computes AP exactly the way `tools/compare.py` does, from
the same function, so the number checked here is the number that reaches the table.

    python tools/check_scorecard.py work/scorecards/temporal_nps-s0.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from benchmarks.scorecard import SequenceResult, pooled_ap  # noqa: E402


def load_sequences(payload: dict) -> list[SequenceResult]:
    """Rebuild SequenceResult objects from a serialised scorecard."""
    out = []
    for s in payload.get("sequences", []):
        out.append(SequenceResult(
            sequence=s.get("sequence", "?"),
            n_gt=int(s.get("n_gt", 0)),
            n_frames=int(s.get("n_frames", 0)),
            conditions=list(s.get("conditions", []) or []),
            detections=[(float(sc), str(o)) for sc, o in (s.get("detections") or [])],
        ))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scorecard", type=Path)
    ap.add_argument("--allow-zero", action="store_true",
                    help="accept AP == 0 as a real result. A genuine zero is possible; "
                         "requiring this flag makes it a stated claim rather than an "
                         "outcome nobody looked at.")
    a = ap.parse_args()

    if not a.scorecard.is_file():
        print(f"ABORT: no scorecard at {a.scorecard}", file=sys.stderr)
        return 2
    payload = json.loads(a.scorecard.read_text(encoding="utf-8"))
    seqs = load_sequences(payload)

    n_frames = sum(s.n_frames for s in seqs)
    n_gt = sum(s.n_gt for s in seqs)
    n_det = sum(len(s.detections) for s in seqs)
    dead = [s.sequence for s in seqs if s.n_frames == 0]
    apv = pooled_ap(seqs) if seqs else 0.0

    print(f"  {payload.get('model','?')} on {payload.get('dataset_key','?')} "
          f"[{payload.get('split','?')}]")
    print(f"  sequences={len(seqs)}  frames={n_frames}  gt={n_gt}  detections={n_det}")
    print(f"  pooled AP = {apv:.4f}")
    if payload.get("git_dirty"):
        print("  NOTE: built from a dirty tree; not reproducible from a commit")

    if not seqs:
        print("ABORT: scorecard has no sequences", file=sys.stderr)
        return 2
    if dead:
        # Per-sequence, not just pooled: a partial resolution failure leaves the other
        # sequences scoring normally, so the pooled AP stays plausibly non-zero while
        # these carry their full ground truth into the denominator with no detections.
        print(f"ABORT: {len(dead)} sequence(s) scored 0 frames -- they are being charged "
              f"as total misses: {dead[:5]}", file=sys.stderr)
        return 2
    if n_det == 0:
        print("ABORT: no detections at all across every sequence -- this is an inference "
              "or path failure, not a model result", file=sys.stderr)
        return 2
    if apv == 0.0 and not a.allow_zero:
        print("ABORT: pooled AP is exactly 0 over a full set of sequences. That is what a "
              "plumbing failure looks like; pass --allow-zero if it is genuinely the "
              "result.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
