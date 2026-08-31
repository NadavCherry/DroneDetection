#!/usr/bin/env python3
"""Prove the test suite really is torch-free, by running it with torch uninstallable.

CI installs only numpy, scipy, opencv-headless and pytest. A top-level
``import torch`` anywhere in a test's import chain therefore passes on a developer
machine — where torch is present — and fails in CI. That failure mode has a long history
in this repo (`.github/workflows/tests.yml` and CLAUDE.md both warn about it), and the
existing defence is a convention: keep such imports lazy, inside the function that needs
them. A convention is not a check.

This is the check. It installs an import blocker ahead of the normal machinery and runs
the suite, so a module-scope ``import torch`` raises loudly *here*, on the machine that
has torch, instead of silently in CI three commits later.

    python tools/check_ci_safety.py                  # run the whole suite blocked
    python tools/check_ci_safety.py dronedet/tests   # a subset

Exit code is the suite's, so it drops straight into a CI job.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: Packages CI does not install. Adding one here makes it a hard error on a test path.
BLOCKED = ("torch", "torchvision", "ultralytics", "tensorrt", "onnxruntime")

_SITECUSTOMIZE = '''
import sys, importlib.abc

BLOCKED = {blocked!r}


class _Blocker(importlib.abc.MetaPathFinder):
    """Raise on import of anything CI will not have installed."""

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in BLOCKED:
            raise ImportError(
                f"CI-SAFETY VIOLATION: '{{fullname}}' was imported on a test path. "
                f"CI installs only numpy/scipy/opencv-headless/pytest, so this passes "
                f"locally and fails there. Move the import inside the function that "
                f"needs it — that is the existing pattern in pursuit/perception.py."
            )
        return None


sys.meta_path.insert(0, _Blocker())
'''


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="*", help="paths to test (default: pytest.ini testpaths)")
    ap.add_argument("--blocked", nargs="*", default=list(BLOCKED))
    a = ap.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "sitecustomize.py").write_text(
            _SITECUSTOMIZE.format(blocked=set(a.blocked)), encoding="utf-8")
        # os.pathsep, not ":" -- on Windows the separator is ";", so a hardcoded
        # colon produces ONE nonsense path entry, PYTHONPATH silently does nothing,
        # the blocking sitecustomize never loads, and the check reports green while
        # testing nothing.
        env_path = os.pathsep.join((tmp, str(REPO)))
        print(f"running the suite with {sorted(a.blocked)} blocked at import ...")
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *a.targets],
            cwd=REPO, env={**_env(), "PYTHONPATH": env_path},
        )
    if proc.returncode == 0:
        print("\nCI-safe: nothing on a test path imports "
              f"{', '.join(sorted(a.blocked))} at module scope.")
    else:
        print("\nNOT CI-safe — see the ImportError above. The fix is to move the import "
              "inside the function that needs it.", file=sys.stderr)
    return proc.returncode


def _env() -> dict:
    import os
    return dict(os.environ)


if __name__ == "__main__":
    raise SystemExit(main())
