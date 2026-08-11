"""Painting a pursuit so a failure can be seen instead of inferred.

A telemetry table says a run ended at 3.4 m. It does not say whether the seeker
was locked onto the drone or onto a cloud, whether the target left the top of
the frame or the side, or whether the last second was a controlled closure or a
tail chase that ran out of time. All of those look identical in a number and
obvious in a picture, so this module exists.

Three panels, each answering a different question:

* **The camera frame** -- what the detector saw and what it decided. Ground
  truth in green, raw detections in red, the tracker's belief in yellow. Where
  those three disagree is the perception's failure mode, spelled out.
* **A magnified inset** -- the target is 14 pixels across at 30 m, which is
  invisible at video scale. The inset is the only way to judge a box by eye.
* **A top-down map** -- both flight paths. Whether the chaser cut the corner or
  followed the target around it is a statement about the guidance law, and it is
  the one thing the camera view can never show.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

GREEN = (80, 220, 90)
RED = (60, 60, 245)
YELLOW = (40, 220, 240)
CYAN = (230, 200, 60)
GREY = (140, 140, 140)
WHITE = (245, 245, 245)
DARK = (28, 28, 32)

MODE_COLOR = {
    "SEARCH": GREY,
    "ACQUIRE": CYAN,
    "PURSUE": GREEN,
    "TERMINAL": (60, 140, 255),
    "REACQUIRE": (60, 200, 255),
    "HIT": (80, 80, 255),
}


def _text(img, s, org, scale=0.5, color=WHITE, thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thick, cv2.LINE_AA)


def _box(img, b, color, thick=1, pad=0):
    if b is None:
        return
    x1, y1, x2, y2 = (int(round(v)) for v in b)
    cv2.rectangle(img, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), color, thick)


class PursuitRecorder:
    """Builds the annotated video one frame at a time.

    Args:
        path: Output ``.mp4``.
        size: ``(width, height)`` of the camera pane before composition.
        fps: Playback rate.
        scale: Downscale applied to the camera pane. 1440x840 at 20 fps is a
            large file for something mostly looked at once.
        arena_m: Half-width of the top-down map, metres.
    """

    def __init__(self, path, size: Tuple[int, int], fps: float = 20.0,
                 scale: float = 0.7, arena_m: float = 110.0) -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.scale = float(scale)
        self.arena_m = float(arena_m)
        self.pane = (int(size[0] * self.scale), int(size[1] * self.scale))
        self.map_px = 260
        self.inset_px = 200
        self.width = self.pane[0] + self.map_px
        self.height = max(self.pane[1], self.map_px + self.inset_px)
        self.writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"),
                                      float(fps), (self.width, self.height))
        self.chaser_path: List[Tuple[float, float]] = []
        self.target_path: List[Tuple[float, float]] = []
        self.centre: Optional[Tuple[float, float]] = None
        self.n = 0

    # -- per frame ----------------------------------------------------------

    def __call__(self, frame, gt, est, gs, perception, chaser, target) -> None:
        if frame is None:
            return
        if self.centre is None:
            self.centre = (chaser.xyz[0], chaser.xyz[1])
        self.chaser_path.append((chaser.xyz[0], chaser.xyz[1]))
        self.target_path.append((target.xyz[0], target.xyz[1]))

        cam = self._camera_pane(frame, gt, est, gs, perception)
        inset = self._inset(frame, gt, est)
        amap = self._map(gs)

        canvas = np.full((self.height, self.width, 3), DARK, np.uint8)
        canvas[:cam.shape[0], :cam.shape[1]] = cam
        canvas[:self.map_px, self.pane[0]:] = amap
        canvas[self.map_px:self.map_px + inset.shape[0], self.pane[0]:] = inset
        self.writer.write(canvas)
        self.n += 1

    def _camera_pane(self, frame, gt, est, gs, perception):
        img = cv2.resize(frame, self.pane, interpolation=cv2.INTER_AREA)
        img = np.ascontiguousarray(img[:, :, ::-1])      # sim renders RGB
        s = self.scale

        for b in perception.last_boxes:
            _box(img, (b.x1 * s, b.y1 * s, b.x2 * s, b.y2 * s), RED, 1, pad=3)

        if gt.get("bbox"):
            _box(img, [v * s for v in gt["bbox"]], GREEN, 1, pad=6)

        if est.valid and est.u is not None:
            u, v = est.u * s, est.v * s
            r = max(10.0, 0.5 * (est.span_px or 20.0) * s + 8)
            colour = YELLOW if est.source == "detector" else CYAN
            cv2.circle(img, (int(u), int(v)), int(r), colour, 1, cv2.LINE_AA)
            cv2.line(img, (int(u - r - 6), int(v)), (int(u - r + 2), int(v)), colour, 1)
            cv2.line(img, (int(u + r - 2), int(v)), (int(u + r + 6), int(v)), colour, 1)

        cx, cy = int(self.pane[0] * 0.5), int(self.pane[1] * 0.5)
        cv2.drawMarker(img, (cx, cy), (90, 90, 90), cv2.MARKER_CROSS, 14, 1)

        self._hud(img, gt, est, gs)
        return img

    def _hud(self, img, gt, est, gs):
        h = img.shape[0]
        colour = MODE_COLOR.get(gs.mode, WHITE)
        cv2.rectangle(img, (0, 0), (img.shape[1], 26), (18, 18, 22), -1)
        _text(img, f"{gs.mode}", (10, 19), 0.62, colour, 2)
        _text(img, f"t={gt.get('t', 0):5.2f}s", (140, 19), 0.5, WHITE)
        rng_t = gt.get("range_m")
        _text(img, f"range true {rng_t:6.2f} m" if rng_t is not None else "range --",
              (250, 19), 0.5, GREEN)
        _text(img, ("est " + (f"{gs.range_est:6.2f} m" if gs.range_est is not None else "--")),
              (430, 19), 0.5, YELLOW)
        _text(img, f"src {est.source}", (560, 19), 0.5,
              YELLOW if est.source == "detector" else CYAN)

        rows = [
            f"LOS rate   {gs.los_rate * 1000:7.2f} mrad/s",
            f"lateral    {gs.lateral_speed:7.2f} m/s",
            f"closing    {gs.closing_speed:7.2f} m/s",
            f"boresight  {gs.boresight_deg:7.2f} deg" if gs.boresight_deg is not None
            else "boresight       -- deg",
            f"cmd v      {gs.command.vx:5.1f} {gs.command.vy:5.1f} {gs.command.vz:5.1f}",
            f"cmd yaw    {gs.command.yaw_rate:7.2f} rad/s",
            f"gt span    {gt.get('span_px') or 0:7.1f} px",
        ]
        y = h - 12 - 16 * len(rows)
        cv2.rectangle(img, (6, y - 14), (250, h - 6), (18, 18, 22), -1)
        for r in rows:
            _text(img, r, (12, y), 0.44, WHITE)
            y += 16
        if gs.note:
            _text(img, gs.note, (12, h - 8), 0.44, colour)

    def _inset(self, frame, gt, est):
        """Magnified crop around the target -- 14 px is nothing at video scale."""
        pane = np.full((self.inset_px, self.map_px, 3), DARK, np.uint8)
        anchor = gt.get("uv") or ([est.u, est.v] if est.valid else None)
        if anchor is None:
            _text(pane, "target not in frame", (14, self.inset_px // 2), 0.45, GREY)
            return pane
        half = 46
        u, v = int(anchor[0]), int(anchor[1])
        h, w = frame.shape[:2]
        x0, y0 = max(0, min(w - 2 * half, u - half)), max(0, min(h - 2 * half, v - half))
        crop = frame[y0:y0 + 2 * half, x0:x0 + 2 * half]
        if crop.size == 0:
            return pane
        k = min(self.map_px / crop.shape[1], (self.inset_px - 18) / crop.shape[0])
        crop = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST)
        crop = np.ascontiguousarray(crop[:, :, ::-1])
        if gt.get("bbox"):
            b = gt["bbox"]
            _box(crop, ((b[0] - x0) * k, (b[1] - y0) * k,
                        (b[2] - x0) * k, (b[3] - y0) * k), GREEN, 1, pad=2)
        if est.valid and est.bbox is not None:
            b = est.bbox
            _box(crop, ((b[0] - x0) * k, (b[1] - y0) * k,
                        (b[2] - x0) * k, (b[3] - y0) * k), YELLOW, 1, pad=4)
        pane[18:18 + crop.shape[0], :crop.shape[1]] = crop
        _text(pane, f"x{k:.1f} inset", (8, 13), 0.4, GREY)
        return pane

    def _map(self, gs):
        """Top-down view: did the chaser cut the corner or follow it round?"""
        pane = np.full((self.map_px, self.map_px, 3), (22, 22, 26), np.uint8)
        if self.centre is None:
            return pane
        k = (self.map_px * 0.5) / self.arena_m
        ox, oy = self.centre

        def to_px(p):
            return (int(self.map_px * 0.5 + (p[0] - ox) * k),
                    int(self.map_px * 0.5 - (p[1] - oy) * k))

        for r in (25, 50, 75, 100):
            if r * k < self.map_px * 0.5:
                cv2.circle(pane, (self.map_px // 2, self.map_px // 2),
                           int(r * k), (38, 38, 44), 1)
        for path, colour in ((self.target_path, GREEN), (self.chaser_path, YELLOW)):
            if len(path) > 1:
                pts = np.array([to_px(p) for p in path], np.int32)
                cv2.polylines(pane, [pts], False, colour, 1, cv2.LINE_AA)
        if self.chaser_path:
            cv2.circle(pane, to_px(self.chaser_path[-1]), 4, YELLOW, -1)
        if self.target_path:
            cv2.circle(pane, to_px(self.target_path[-1]), 4, GREEN, -1)
            if self.chaser_path:
                cv2.line(pane, to_px(self.chaser_path[-1]),
                         to_px(self.target_path[-1]), (70, 70, 80), 1)
        _text(pane, "top-down", (8, 14), 0.4, GREY)
        _text(pane, "chaser", (8, self.map_px - 20), 0.4, YELLOW)
        _text(pane, "target", (8, self.map_px - 6), 0.4, GREEN)
        return pane

    def close(self) -> None:
        self.writer.release()


ORANGE = (40, 150, 250)
BLUE = (200, 130, 60)


class RingRecorder:
    """The four-camera engagement, painted so a failure can be seen.

    A ring is harder to read than a single camera and easier to get wrong, so
    the video is built to answer the questions that are specific to it:

    * **Which camera has it, and did the handover work?** All four panes are
      shown, always, in mount order, and the one that currently owns the target
      is outlined. A track that dies at a seam is obvious here and invisible in
      a telemetry table.
    * **Did the ring see it at all, or only the ground truth?** Green is the
      simulator's own label, orange is the appearance detector, red is a motion
      blob, yellow is what the tracker believes. Long-range acquisition is the
      stretch where only red and green exist.
    * **Is the interceptor going to get there first?** The map draws the
      defended structures as real footprints and the intruder's aim point as a
      cross, so "arriving late" -- the failure the earlier suites could not even
      express -- is something you can watch happen.
    """

    def __init__(self, path, ring, fps: float = 20.0, pane_w: int = 640,
                 arena_m: float = 240.0, buildings: Optional[Sequence[dict]] = None,
                 aim_xy: Optional[Sequence[float]] = None,
                 strike_radius_m: float = 6.0) -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.ring = ring
        self.names = [c.name for c in ring.cameras]
        intr = ring.cameras[0].intr
        self.scale = pane_w / intr.width
        self.pane = (pane_w, max(1, int(round(intr.height * self.scale))))
        self.cols = 2
        self.rows = (len(self.names) + self.cols - 1) // self.cols
        self.side = 320
        self.map_px = self.side
        self.inset_px = 240
        self.hud_h = 26
        # The grid starts *below* the status bar. Drawn over it, the bar covers
        # the top row's camera names -- and with four near-identical panes of
        # town, "which camera is this" is the one label the viewer cannot
        # reconstruct.
        self.width = self.pane[0] * self.cols + self.side
        self.height = (self.hud_h
                       + max(self.pane[1] * self.rows, self.map_px + self.inset_px))
        self.writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"),
                                      float(fps), (self.width, self.height))
        self.arena_m = float(arena_m)
        self.buildings = list(buildings or [])
        self.aim_xy = tuple(aim_xy) if aim_xy is not None else None
        self.strike_radius_m = float(strike_radius_m)
        self.chaser_path: List[Tuple[float, float]] = []
        self.target_path: List[Tuple[float, float]] = []
        self.centre: Optional[Tuple[float, float]] = None
        self.n = 0

    def __call__(self, frames, gt, est, gs, perception, chaser, target) -> None:
        if not isinstance(frames, dict) or not frames:
            return
        if self.centre is None:
            self.centre = (chaser.xyz[0], chaser.xyz[1])
        self.chaser_path.append((chaser.xyz[0], chaser.xyz[1]))
        self.target_path.append((target.xyz[0], target.xyz[1]))

        canvas = np.full((self.height, self.width, 3), DARK, np.uint8)
        per_cam = (gt.get("per_camera") or {})
        by_cam = getattr(perception, "last_by_camera", {}) or {}
        dets = getattr(perception, "last_dets", []) or []
        for i, name in enumerate(self.names):
            pane = self._camera_pane(name, frames.get(name), per_cam.get(name),
                                     by_cam.get(name, []), dets, est)
            r, c = divmod(i, self.cols)
            y0 = self.hud_h + r * self.pane[1]
            x0 = c * self.pane[0]
            canvas[y0:y0 + pane.shape[0], x0:x0 + pane.shape[1]] = pane

        x0 = self.pane[0] * self.cols
        top = self.hud_h
        canvas[top:top + self.map_px, x0:] = self._map(chaser, target)
        inset = self._inset(frames, gt, est)
        canvas[top + self.map_px:top + self.map_px + inset.shape[0], x0:] = inset
        self._hud(canvas, gt, est, gs)
        self.writer.write(canvas)
        self.n += 1

    def _camera_pane(self, name, frame, view, boxes, dets, est):
        pane = np.full((self.pane[1], self.pane[0], 3), (16, 16, 20), np.uint8)
        owner = bool(est.valid and est.camera == name)
        if frame is not None:
            img = cv2.resize(frame, self.pane, interpolation=cv2.INTER_AREA)
            pane[:] = np.ascontiguousarray(img[:, :, ::-1])   # the sim renders RGB
        s = self.scale
        if view and view.get("bbox"):
            _box(pane, [v * s for v in view["bbox"]], GREEN, 1, pad=7)
        for b in boxes:
            _box(pane, (b.x1 * s, b.y1 * s, b.x2 * s, b.y2 * s), ORANGE, 1, pad=4)
        cam = self.ring.get(name)
        for d in dets:
            if d.camera != name or d.kind != "motion":
                continue
            r = max(5, int(0.5 * d.span_rad * cam.intr.fx * s) + 4)
            cv2.circle(pane, (int(d.u * s), int(d.v * s)), r, RED, 1, cv2.LINE_AA)
        if owner and est.u is not None:
            r = max(9, int(0.5 * (est.span_px or 20.0) * s) + 7)
            colour = YELLOW if est.source == "detector" else CYAN
            cv2.circle(pane, (int(est.u * s), int(est.v * s)), r, colour, 1,
                       cv2.LINE_AA)
        if owner:
            cv2.rectangle(pane, (0, 0), (self.pane[0] - 1, self.pane[1] - 1),
                          YELLOW, 2)
        cv2.rectangle(pane, (0, 0), (86, 18), (18, 18, 22), -1)
        _text(pane, name, (6, 14), 0.46, YELLOW if owner else GREY)
        return pane

    def _map(self, chaser, target):
        pane = np.full((self.map_px, self.map_px, 3), (22, 22, 26), np.uint8)
        if self.centre is None:
            return pane
        k = (self.map_px * 0.5) / self.arena_m
        ox, oy = self.centre

        def to_px(p):
            return (int(self.map_px * 0.5 + (p[0] - ox) * k),
                    int(self.map_px * 0.5 - (p[1] - oy) * k))

        for r in (50, 100, 150, 200):
            if r * k < self.map_px * 0.55:
                cv2.circle(pane, (self.map_px // 2, self.map_px // 2),
                           int(r * k), (38, 38, 44), 1)
        for b in self.buildings:
            cx, cy = b["xy"]
            hw, hd = b["footprint_m"][0] / 2.0, b["footprint_m"][1] / 2.0
            p1, p2 = to_px((cx - hw, cy + hd)), to_px((cx + hw, cy - hd))
            cv2.rectangle(pane, p1, p2, BLUE, 1)
        if self.aim_xy is not None:
            p = to_px(self.aim_xy)
            cv2.drawMarker(pane, p, RED, cv2.MARKER_TILTED_CROSS, 11, 2)
            cv2.circle(pane, p, max(2, int(self.strike_radius_m * k)), RED, 1)
        for path, colour in ((self.target_path, GREEN), (self.chaser_path, YELLOW)):
            if len(path) > 1:
                cv2.polylines(pane, [np.array([to_px(p) for p in path], np.int32)],
                              False, colour, 1, cv2.LINE_AA)
        if self.chaser_path:
            cv2.circle(pane, to_px(self.chaser_path[-1]), 4, YELLOW, -1)
        if self.target_path:
            cv2.circle(pane, to_px(self.target_path[-1]), 4, GREEN, -1)
        _text(pane, "top-down", (8, 14), 0.4, GREY)
        _text(pane, "interceptor", (8, self.map_px - 34), 0.4, YELLOW)
        _text(pane, "intruder", (8, self.map_px - 20), 0.4, GREEN)
        _text(pane, "buildings / aim", (8, self.map_px - 6), 0.4, BLUE)
        return pane

    def _inset(self, frames, gt, est):
        """Magnified, and contrast-stretched, because 3 px of drone against
        bright sky is invisible at any magnification without it.

        The whole point of this pane is to let a human confirm by eye what the
        numbers claim, and a target whose entire signature is being slightly
        darker than the sky behind it does not survive being shown at native
        contrast in a 300-pixel box. The stretch is per-crop and local, so it
        says nothing about the detector's input -- which is the unstretched
        frame -- and everything about whether the thing is really there.
        """
        pane = np.full((self.inset_px, self.side, 3), DARK, np.uint8)
        name = gt.get("camera") or (est.camera if est.valid else None)
        frame = frames.get(name) if name else None
        anchor = gt.get("uv") or gt.get("analytic_uv") or (
            [est.u, est.v] if est.valid and est.u is not None else None)
        if frame is None or anchor is None:
            _text(pane, "target not in any frame", (12, self.inset_px // 2),
                  0.45, GREY)
            return pane
        half = 26
        h, w = frame.shape[:2]
        u, v = int(anchor[0]), int(anchor[1])
        x0 = max(0, min(w - 2 * half, u - half))
        y0 = max(0, min(h - 2 * half, v - half))
        crop = frame[y0:y0 + 2 * half, x0:x0 + 2 * half]
        if crop.size == 0:
            return pane
        kk = min(self.side / crop.shape[1], (self.inset_px - 18) / crop.shape[0])
        crop = cv2.resize(crop, None, fx=kk, fy=kk, interpolation=cv2.INTER_NEAREST)
        crop = np.ascontiguousarray(crop[:, :, ::-1])
        lo, hi = float(crop.min()), float(crop.max())
        if hi - lo > 4.0:
            crop = np.clip((crop.astype(np.float32) - lo) * (255.0 / (hi - lo)),
                           0, 255).astype(np.uint8)
        view = (gt.get("per_camera") or {}).get(name) or gt
        if view.get("bbox"):
            b = view["bbox"]
            _box(crop, ((b[0] - x0) * kk, (b[1] - y0) * kk,
                        (b[2] - x0) * kk, (b[3] - y0) * kk), GREEN, 1, pad=2)
        pane[18:18 + crop.shape[0], :crop.shape[1]] = crop
        _text(pane, f"x{kk:.1f}  {name}  (contrast stretched)", (8, 13), 0.38, GREY)
        return pane

    def _hud(self, canvas, gt, est, gs):
        h = self.height
        colour = MODE_COLOR.get(gs.mode, WHITE)
        cv2.rectangle(canvas, (0, 0), (self.width, self.hud_h), (18, 18, 22), -1)
        _text(canvas, gs.mode, (10, 18), 0.6, colour, 2)
        _text(canvas, f"t={gt.get('t', 0):5.2f}s", (150, 18), 0.5, WHITE)
        rng_t = gt.get("range_m")
        _text(canvas, f"range {rng_t:6.1f} m" if rng_t is not None else "range --",
              (260, 18), 0.5, GREEN)
        _text(canvas, "est " + (f"{gs.range_est:6.1f} m"
                                if gs.range_est is not None else "--"),
              (420, 18), 0.5, YELLOW)
        _text(canvas, f"span {gt.get('span_px') or 0:5.1f} px", (560, 18), 0.5, WHITE)
        _text(canvas, f"src {est.source}/{est.kind}", (720, 18), 0.5,
              YELLOW if est.source == "detector" else CYAN)
        seen = ",".join(gt.get("seen_by") or []) or "-"
        _text(canvas, f"seen by {seen}", (900, 18), 0.5, GREY)
        if gs.note:
            _text(canvas, gs.note, (12, h - 8), 0.46, colour)

    def close(self) -> None:
        self.writer.release()
