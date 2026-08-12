"""Dataset adapters: the only code that knows a dataset's native format.

Everything downstream sees two canonical things per sequence and nothing else:

* a **`dronedet` ground-truth json** — boxes in ``(cx, cy, w, h)`` centre format, true
  extent, non-target classes preserved as ``ignore`` distractors;
* an **`ImageSource`** — a video to decode sequentially, or a list of stills.

`tools/prepare_data.py` turns those into `gt/` plus a YOLO training set that honours the
dataset's official split. Adding a dataset means adding one file here.

Registry, not a plugin scan: an import that fails is louder than a dataset that quietly
does not appear.

    from benchmarks.adapters import build
    ad = build("ardmav")                     # default root from the repo layout
    ad = build("yolo_dir", root="path/to/x") # generic; root required
"""

from __future__ import annotations

from pathlib import Path

from .ardmav import ArdMavAdapter
from .base import (
    DRONE_ALIASES,
    Adapter,
    Box,
    ImageSource,
    image_size,
    parse_yolo_label,
    read_class_names,
)
from .halmstad import HalmstadAdapter
from .uav_smid import UavSmidAdapter
from .yolo_dir import YoloDirAdapter

REPO = Path(__file__).resolve().parents[2]

ADAPTERS: dict[str, type[Adapter]] = {
    "ardmav": ArdMavAdapter,
    "halmstad": HalmstadAdapter,
    "uav_smid": UavSmidAdapter,
    "yolo_dir": YoloDirAdapter,
}

#: Where each dataset lands under `data/external/` when fetched. These are defaults for
#: convenience only -- every entry point takes an explicit `--root`, because the download
#: is 14.6 GB for ARD-MAV alone and will often live on another disk.
DEFAULT_ROOTS: dict[str, Path] = {
    "ardmav": REPO / "data/external/ard_mav/ARD-MAV",
    "halmstad": REPO / "data/external/halmstad",
    "uav_smid": REPO / "data/external/uav_smid",
}


def build(key: str, root: str | Path | None = None, **kwargs) -> Adapter:
    """Construct an adapter by catalog key.

    `root` falls back to `DEFAULT_ROOTS`; `yolo_dir` has no default because it describes
    a layout rather than a dataset, and inventing a location for it would only produce a
    confusing FileNotFoundError somewhere else.
    """
    if key not in ADAPTERS:
        raise KeyError(f"unknown adapter {key!r}; have {sorted(ADAPTERS)}")
    if root is None:
        root = DEFAULT_ROOTS.get(key)
        if root is None:
            raise ValueError(f"adapter {key!r} has no default root -- pass root=")
    return ADAPTERS[key](root, **kwargs)


__all__ = [
    "ADAPTERS", "DEFAULT_ROOTS", "build",
    "Adapter", "Box", "ImageSource", "DRONE_ALIASES",
    "ArdMavAdapter", "HalmstadAdapter", "UavSmidAdapter", "YoloDirAdapter",
    "image_size", "parse_yolo_label", "read_class_names",
]
