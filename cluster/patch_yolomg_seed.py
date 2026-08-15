#!/usr/bin/env python3
"""Give YOLOMG's trainer a --seed flag. Idempotent; run before every training job.

WHY THIS PATCH EXISTS, AND WHY IT IS NOT CHEATING
-------------------------------------------------
`tools/sota` forbids modifying a competitor's method. This does not touch the method: it
touches the random number generator, and it is required for the campaign to be honest
rather than optional for convenience.

Their train.py seeds with:

    init_seeds(1 + RANK)

RANK is -1 on a single GPU, so every run seeds identically. Three "seeds" would have been
three byte-identical runs. We would then have reported a mean and a confidence interval
over a sample whose variance is zero by construction -- a fabricated interval, and a far
worse outcome than the crash that exposed this (`train.py: error: unrecognized arguments:
--seed 0`). Our own arm varies its seed properly, so leaving theirs fixed would also mean
the two arms' variance estimates were not measuring the same thing.

WHAT IT CHANGES: two lines. The argument is added, and it is threaded into the existing
init_seeds call. Nothing about the architecture, loss, data, or schedule moves.

IDEMPOTENT: safe to run before every job in an array; a second run is a no-op. It verifies
its own result and exits non-zero if the file is not in the expected shape, so a job never
proceeds to training against a half-patched or upstream-changed trainer.
"""

from __future__ import annotations

import sys
from pathlib import Path

TRAIN_PY = (Path(__file__).resolve().parent.parent
            / "third_party" / "YOLOMG" / "train.py")

OLD_SEED_CALL = "init_seeds(1 + RANK)"
NEW_SEED_CALL = "init_seeds(opt.seed + 1 + RANK)"

ANCHOR = ("    parser.add_argument('--bbox_interval', type=int, default=-1, "
          "help='W&B: Set bounding-box image logging interval')")
NEW_ARG = ("    parser.add_argument('--seed', type=int, default=0, "
           "help='RNG seed; added by SpeckLock so 3 seeds are 3 different runs')")


def main() -> int:
    if not TRAIN_PY.is_file():
        print(f"ABORT: {TRAIN_PY} not found", file=sys.stderr)
        return 1
    src = TRAIN_PY.read_text(encoding="utf-8", errors="replace")

    already_arg = "'--seed'" in src
    already_call = NEW_SEED_CALL in src
    if already_arg and already_call:
        print("already patched; nothing to do")
        return 0

    if not already_arg:
        if ANCHOR not in src:
            print("ABORT: could not find the argparse anchor line; upstream train.py has "
                  "changed shape and this patch must be re-checked by hand", file=sys.stderr)
            return 1
        src = src.replace(ANCHOR, ANCHOR + "\n" + NEW_ARG, 1)

    if not already_call:
        if src.count(OLD_SEED_CALL) != 1:
            print(f"ABORT: expected exactly one {OLD_SEED_CALL!r}, found "
                  f"{src.count(OLD_SEED_CALL)}", file=sys.stderr)
            return 1
        src = src.replace(OLD_SEED_CALL, NEW_SEED_CALL, 1)

    TRAIN_PY.write_text(src, encoding="utf-8")

    check = TRAIN_PY.read_text(encoding="utf-8", errors="replace")
    if "'--seed'" not in check or NEW_SEED_CALL not in check:
        print("ABORT: patch did not verify after writing", file=sys.stderr)
        return 1
    print(f"patched {TRAIN_PY}: --seed added, threaded into init_seeds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
