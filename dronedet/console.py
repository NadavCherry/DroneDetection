"""Make stdout/stderr carry the characters this project actually prints.

Python picks the console encoding from the locale, which on this machine is **cp1255**
(Hebrew Windows). Almost every table in this repo prints characters cp1255 has no code
point for -- `IoU≥0.5` from `Protocol.describe`, `τ=12` from the centre-distance protocol,
`⚠`/`✅`/`❌` in the comparison tables, `±` beside every interval -- so on this machine:

    UnicodeEncodeError: 'charmap' codec can't encode character '\\u2265'

and the tool dies *after* doing the work, while printing the answer. Not a cosmetic
problem: `tools/compare.py` is the last stage of an unattended run, so the crash lands
after the GPU-hours are spent and destroys the one artifact they were spent on.

Distinct from the file-encoding fix. `read_text`/`write_text` now pass encoding="utf-8"
everywhere, which fixed what this project *stores*; this fixes what it *shows*.

`errors="replace"` on purpose: a mangled character in a table is a nuisance, and a dead
process at the end of an overnight run is not.
"""
from __future__ import annotations

import sys


def use_utf8_stdio() -> None:
    """Reconfigure stdout/stderr to UTF-8. Safe to call more than once, and on streams
    that have been redirected to something without `reconfigure` (a pytest capture
    buffer, a pipe wrapper), where it does nothing rather than raising."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):        # already detached, or not a real text stream
            pass
