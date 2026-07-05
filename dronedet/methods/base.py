"""Method interface and the shared video-processing runner."""

from __future__ import annotations

import time

import numpy as np

from ..detections import DetectionSet
from ..stabilize import Stabilizer, shift_of
from ..video import frames, probe


class BaseMethod:
    """A detector consuming one frame at a time.

    ``process`` receives the original BGR frame and the 2x3 stabilization
    matrix mapping it into reference (frame 0) coordinates, and returns
    detections in *original frame* coordinates.
    """

    name: str = "base"

    def process(self, idx: int, frame_bgr: np.ndarray, m_stab: np.ndarray):
        raise NotImplementedError

    def close(self) -> None:
        pass


def run_method(
    video_path: str,
    method: BaseMethod,
    stop: int | None = None,
    stab_mode: str = "translation",
    progress_every: int = 100,
    stab_scale: float = 1.0,
    det_stride: int = 1,
) -> DetectionSet:
    """stab_scale < 1 estimates the camera transform on a downscaled frame (cheap).
    det_stride > 1 runs the detector on every Nth frame only (the tracker coasts
    between) -- valid for STATELESS detectors; motion methods need every frame."""
    info = probe(video_path)
    stab = Stabilizer(stab_mode, scale=stab_scale)
    ds = DetectionSet(video=video_path, method=method.name)
    shifts: dict[str, list[float]] = {}
    transforms: dict[str, list[float]] = {}   # full 2x3 current->reference (affine-aware tracking)
    t0 = time.perf_counter()
    n = 0
    for idx, frame in frames(video_path, stop=stop):
        m = stab.update(frame)
        dets = method.process(idx, frame, m) if idx % det_stride == 0 else []
        ds.add(idx, dets)
        dx, dy = shift_of(m)
        shifts[str(idx)] = [round(dx, 3), round(dy, 3)]
        transforms[str(idx)] = [round(float(v), 6) for v in np.asarray(m).reshape(-1)]
        n += 1
        if progress_every and n % progress_every == 0:
            el = time.perf_counter() - t0
            print(f"  [{method.name}] frame {idx} ({n / el:.1f} fps)", flush=True)
    elapsed = time.perf_counter() - t0
    ds.meta = {
        "fps_end_to_end": round(n / elapsed, 2),
        "n_frames": n,
        "video_fps": info.fps,
        "video_size": [info.width, info.height],
        "stab_mode": stab_mode,
        "shifts": shifts,
        "transforms": transforms,
    }
    method.close()
    return ds
