"""The wire between the brain and the simulator.

Isaac Sim lives in a container that has no ultralytics, no scipy and none of this
project's weights; the perception stack lives on the host and cannot import
Isaac. They share one thing -- ``/tmp/dev`` in the container is a directory on
the host (``~/isaac_dev_root`` by default; set ``DRONEDET_SIM_ROOT`` to move it)
-- so a unix socket in that directory is visible to both processes and this
module is the only code both sides run.

Why a socket and not a script that does everything: booting Isaac Sim with a
scene loaded costs 30-60 seconds, and the brain is the half that changes every
few minutes. With the simulator as a long-lived *server*, a brain restart costs
the two seconds it takes to load a YOLO checkpoint, and the same booted sim
serves an entire scenario matrix -- data collection, one pursuit, then fifty
more -- without ever reloading the town.

Frames are sent **raw**, not JPEG. The target is 10-30 pixels across and the
detector keys on exactly the kind of small local contrast a JPEG quantiser
throws away first; 3.6 MB a frame over a unix socket is far cheaper than
arguing about whether a miss was the algorithm or the codec.

Message format, both directions::

    <u32 header_len><header: utf-8 JSON><u32 payload_len><payload: raw bytes>

The header always decodes to a dict. ``payload_len`` is 0 when there is no
payload; an RGB frame arrives as ``H*W*3`` uint8 with its shape in the header.
"""
from __future__ import annotations

import json
import os
import socket
import struct
from typing import Any, Optional, Tuple

DEFAULT_SOCKET = "/tmp/dev/pursuit/sim.sock"
"""Container-side path -- fixed, because the bind mount inside the container is."""


def host_socket() -> str:
    """The same socket as seen from the *host*, where the brain runs.

    Resolved rather than hardcoded so a clone works on someone else's machine:
    ``$DRONEDET_SIM_SOCK`` wins outright, else ``$DRONEDET_SIM_ROOT/pursuit/sim.sock``,
    else ``~/isaac_dev_root/pursuit/sim.sock``. Whatever this returns must be the
    host side of the bind mount that puts ``/tmp/dev`` inside the container --
    the two processes are talking through one inode, not two paths that look
    alike.
    """
    explicit = os.environ.get("DRONEDET_SIM_SOCK")
    if explicit:
        return explicit
    root = os.environ.get("DRONEDET_SIM_ROOT") or os.path.join(
        os.path.expanduser("~"), "isaac_dev_root")
    return os.path.join(root, "pursuit", "sim.sock")

_HDR = struct.Struct("!I")


def send_msg(sock: socket.socket, header: dict, payload: bytes = b"") -> None:
    """Send one header+payload message."""
    blob = json.dumps(header).encode("utf-8")
    sock.sendall(_HDR.pack(len(blob)) + blob + _HDR.pack(len(payload)))
    if payload:
        sock.sendall(payload)


def recv_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly ``n`` bytes, or raise ``ConnectionError`` if the peer went away.

    ``recv`` is free to return short reads on a stream socket and does so
    routinely once a message is bigger than the kernel buffer -- which a 3.6 MB
    frame always is. Every read here goes through this function for that reason.
    """
    chunks = []
    got = 0
    while got < n:
        b = sock.recv(min(1 << 20, n - got))
        if not b:
            raise ConnectionError(f"peer closed after {got}/{n} bytes")
        chunks.append(b)
        got += len(b)
    return b"".join(chunks)


def recv_msg(sock: socket.socket) -> Tuple[dict, bytes]:
    """Receive one header+payload message."""
    (hlen,) = _HDR.unpack(recv_exact(sock, 4))
    header = json.loads(recv_exact(sock, hlen).decode("utf-8"))
    (plen,) = _HDR.unpack(recv_exact(sock, 4))
    payload = recv_exact(sock, plen) if plen else b""
    return header, payload


def frame_from_payload(header: dict, payload: bytes):
    """Rebuild the ``(H, W, 3)`` uint8 array a ``step``/``reset`` reply carries.

    Takes the *first* frame. A payload can hold several -- a camera ring sends
    four, and a split-view run appends the target's camera -- so this slices to
    the declared shape rather than reshaping the whole buffer. Reshaping the lot
    is what a single-camera client would do, and it would start raising the day
    somebody turned a second camera on, which is the wrong failure: the first
    frame is still exactly what it asked for.
    """
    import numpy as np

    shape = header.get("frame_shape")
    if not shape or not payload:
        return None
    n = int(np.prod(shape))
    return np.frombuffer(payload[:n], dtype=np.uint8).reshape(tuple(shape))


def frames_from_payload(header: dict, payload: bytes) -> dict:
    """Split a ring reply into ``{camera name: (H, W, 3) uint8}``.

    The wire carries the frames concatenated in the order ``header["cameras"]``
    lists them, and that order is the simulator's mount order -- which is why
    the order is part of the protocol rather than something both sides sort
    independently. A ring whose frames are one camera out of step is a system
    that steers 90 degrees away from its target and looks entirely healthy doing
    it.
    """
    import numpy as np

    out = {}
    off = 0
    for cam in header.get("cameras") or []:
        shape = cam.get("shape")
        if not shape:
            continue
        n = int(np.prod(shape))
        if off + n > len(payload):
            break
        out[cam["name"]] = np.frombuffer(payload[off:off + n],
                                         dtype=np.uint8).reshape(tuple(shape))
        off += n
    return out


class SimClient:
    """Brain-side handle on the simulator.

    Args:
        path: Socket path as the *client* sees it. On the host, that is what
            :func:`host_socket` returns; inside the container it is
            :data:`DEFAULT_SOCKET`.
        timeout_s: Per-call socket timeout. Generous by default: a ``reset`` that
            has to warm the render pipeline can take several seconds.
    """

    def __init__(self, path: str, timeout_s: float = 120.0) -> None:
        self.path = str(path)
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(timeout_s)
        self.sock.connect(self.path)

    def call_raw(self, cmd: str, **kw) -> Tuple[dict, bytes]:
        """Send one command and return ``(reply_header, raw_payload)``."""
        send_msg(self.sock, {"cmd": cmd, **kw})
        header, payload = recv_msg(self.sock)
        if header.get("error"):
            raise RuntimeError(f"sim error on {cmd!r}: {header['error']}")
        return header, payload

    def call(self, cmd: str, **kw) -> Tuple[dict, Any]:
        """Send one command and return ``(reply_header, frame_or_None)``."""
        header, payload = self.call_raw(cmd, **kw)
        return header, frame_from_payload(header, payload)

    def call_frames(self, cmd: str, **kw) -> Tuple[dict, dict]:
        """Send one command and return ``(reply_header, {camera: frame})``."""
        header, payload = self.call_raw(cmd, **kw)
        return header, frames_from_payload(header, payload)

    def info(self) -> dict:
        """Camera intrinsics, scene geometry and the target's physical span."""
        return self.call("info")[0]

    def reset(self, chaser: dict, target: dict, settle: int = 4):
        """Place both aircraft and render a settled first frame."""
        return self.call("reset", chaser=chaser, target=target, settle=settle)

    def step(self, chaser: dict, target: dict):
        """Place both aircraft, advance one capture interval, return the frame."""
        return self.call("step", chaser=chaser, target=target)

    def reset_all(self, chaser: dict, target: dict, settle: int = 4):
        """:meth:`reset`, returning every camera's frame."""
        return self.call_frames("reset", chaser=chaser, target=target,
                                settle=settle)

    def step_all(self, chaser: dict, target: dict):
        """:meth:`step`, returning every camera's frame."""
        return self.call_frames("step", chaser=chaser, target=target)

    def close(self) -> None:
        try:
            send_msg(self.sock, {"cmd": "bye"})
        except OSError:
            pass
        self.sock.close()

    def __enter__(self) -> "SimClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
