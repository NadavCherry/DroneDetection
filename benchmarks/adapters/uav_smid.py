"""UAV_SMID — 13,928 stills, five deliberately balanced classes, and its real job.

Mendeley, CC BY 4.0, direct download: helicopter / bomb / drone / bird / aeroplane, roughly
3,200–3,400 objects each. The balance is the point. Almost every anti-UAV training set is
drone-only, so the detectors trained on them have **never been shown a bird** and their
"false positives" are unattributed. This corpus is the cheapest way to change that.

Two things it is not
--------------------
**It is not evidence for the temporal claim.** These are stills. This project's headline
mechanism is a three-moment stabilised stack; a single frame cannot exercise it, and a
number from here must never be quoted as though it did. It belongs to the single-frame
branch and to the loss/matching ablations.

**It is not a five-class detector's training set.** For drone training the four non-drone
classes are *negatives*, not classes:

* The **YOLO labels keep only drone boxes.** A bird image therefore becomes an image with
  an empty label file — which is exactly what teaches "bird = background". Adding a bird
  class instead would teach the detector to *locate* birds, spending capacity on a class
  nobody asked for, and would leave the drone/bird confusion unmeasured because a
  bird-labelled prediction is no longer a false alarm.
* The **ground truth keeps every non-drone box as an ``ignore`` object.** `dronedet.metrics`
  scores a detection on an ignore object as a *distractor*: excluded from precision by
  default but always counted and reported. That is the only route by which "N hits on
  3,162 labelled bird instances" becomes a published number rather than a silence.

Both behaviours come from the base class's `positive_classes` — the split of the class
table into target and distractor happens in exactly one place.

Layout: whatever the release ships (`YoloDirAdapter` discovers it). The class table is read
from the release's own ``data.yaml``/``classes.txt`` and never assumed: the catalog lists
the five names, but the *index order* is a property of the download, and an index order
guessed wrong silently swaps drone and bird — which would invert the one claim this
dataset exists to support.
"""

from __future__ import annotations

from pathlib import Path

from ..catalog import Condition
from .yolo_dir import YoloDirAdapter


class UavSmidAdapter(YoloDirAdapter):
    key = "uav_smid"

    #: Only the drone is a target. 'bomb' joins bird/aeroplane/helicopter as a distractor:
    #: it is an air-object class this project makes no claim about, and the safe direction
    #: to fail in is "not a drone".
    positive_classes = frozenset({"drone", "uav", "mav", "quadcopter"})

    def __init__(self, root: str | Path, *, names: dict[int, str] | None = None) -> None:
        super().__init__(root, names=names)

    def conditions(self, seq: str) -> tuple[Condition, ...]:
        """Empty on purpose.

        The catalog records the corpus as `Condition.CLEAR`, but the release ships no
        per-image weather or illumination metadata, so no *image* here can be attributed
        to a condition. Returning the corpus-level label per sequence would put an
        unverified condition into a stratified table; read `DATASETS['uav_smid'].conditions`
        when the corpus-level answer is what is wanted.
        """
        return ()

    def distractor_classes(self) -> list[str]:
        """The non-target class names actually present in this download — for the manifest,
        so a report can say which distractors a false-alarm rate was measured against
        rather than repeating the catalog's list from memory."""
        return sorted(n for n in self.class_names().values()
                      if n.lower() not in self.positive_classes)
