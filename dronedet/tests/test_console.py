"""stdout must be able to carry the characters this project prints.

Not cosmetic. `tools/compare.py` is the last stage of an unattended run and prints
`Protocol.describe()`, which contains a U+2265. On this machine the console encoding is
cp1255, which has no code point for it, so the process died with UnicodeEncodeError
*after* the GPU-hours were spent and while printing the answer they were spent on.
"""
from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

from benchmarks.protocol import ARDMAV_OFFICIAL, SPECKLOCK_CENTRE
from dronedet.console import use_utf8_stdio

REPO = Path(__file__).resolve().parents[2]


def test_the_protocol_descriptions_really_do_contain_non_ascii():
    """The premise. If these ever become ASCII this whole file is obsolete."""
    assert "≥" in ARDMAV_OFFICIAL.describe()      # IoU>=0.5
    assert "τ" in SPECKLOCK_CENTRE.describe()     # tau=12


def test_cp1255_cannot_encode_them():
    """Establishes that the bug is real on this locale rather than hypothetical."""
    try:
        ARDMAV_OFFICIAL.describe().encode("cp1255")
    except UnicodeEncodeError:
        return
    raise AssertionError("cp1255 encoded U+2265; the premise of this file has changed")


def test_use_utf8_stdio_is_safe_on_a_stream_without_reconfigure():
    """pytest replaces sys.stdout with a capture object; a helper that assumes a real
    TextIOWrapper would explode inside the test suite it is meant to protect."""
    saved = sys.stdout
    try:
        sys.stdout = io.StringIO()             # no .reconfigure
        use_utf8_stdio()                       # must not raise
    finally:
        sys.stdout = saved


def test_use_utf8_stdio_is_idempotent():
    use_utf8_stdio()
    use_utf8_stdio()


def test_a_cli_prints_a_non_ascii_protocol_without_dying():
    """End to end, in a real subprocess, with the console encoding forced to cp1255.

    A unit test cannot catch this: the failure happens in the interpreter's stdout
    encoder, so it only appears in a process whose stdout is genuinely non-UTF-8.
    """
    script = (
        "import sys; sys.path.insert(0, r'%s')\n"
        "from dronedet.console import use_utf8_stdio\n"
        "use_utf8_stdio()\n"
        "from benchmarks.protocol import ARDMAV_OFFICIAL\n"
        "print(ARDMAV_OFFICIAL.describe())\n" % REPO
    )
    p = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                       encoding="utf-8", env={**__import__("os").environ,
                                              "PYTHONIOENCODING": "cp1255"})
    assert p.returncode == 0, f"CLI died printing a protocol: {p.stderr[-500:]}"
    assert "UnicodeEncodeError" not in (p.stderr or "")
