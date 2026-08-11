"""Let ultralytics train on >3-channel images stored as .npy.

ultralytics already has an internal multi-channel path -- `load_image` checks the
loaded array's channel count against the dataset's `channels` and BaseDataset sets
`channels` from data.yaml's `channels:` key. What's missing is only the plumbing
to let a `.npy` file be treated as a *source* image:

  1. `.npy` isn't in IMG_FORMATS, so `get_img_files` filters it out.
  2. the label/image verifier (`check_image`) PIL-opens the file, which fails on
     a raw `.npy` array.

`enable_multichannel()` patches exactly those two spots (idempotent). Everything
downstream -- letterbox, mosaic, perspective warp, flips -- is cv2/numpy and
already channel-agnostic (up to 4 ch). HSV augmentation must be OFF (it BGR->HSV
converts and assumes 3 ch); callers pass hsv 0 0 0.
"""
from __future__ import annotations

_PATCHED = False


def enable_multichannel() -> None:
    global _PATCHED
    if _PATCHED:
        return
    import numpy as np
    import ultralytics.data.utils as U

    U.IMG_FORMATS.add("npy")  # shared set object -> visible to get_img_files everywhere

    _orig_check_image = U.check_image

    def check_image(im_file):
        if str(im_file).lower().endswith(".npy"):
            arr = np.load(im_file, mmap_mode="r")
            h, w = int(arr.shape[0]), int(arr.shape[1])
            assert h > 9 and w > 9, f"image size {(h, w)} <10 pixels"
            return "", (h, w)
        return _orig_check_image(im_file)

    U.check_image = check_image  # verify_image_label resolves check_image from this module
    _PATCHED = True
    print("[mc_data] multi-channel .npy training enabled", flush=True)
