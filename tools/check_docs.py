#!/usr/bin/env python3
"""Check that the documentation still works for a stranger on github.com.

The failure this exists to prevent is specific and was real: a document links to
an artifact under ``work/``, the artifact is gitignored, and the link is a 404
for everyone except the author -- who sees it resolve locally and never notices.
So "does the file exist" is not the test. **"Is it in the index"** is.

    .venv/bin/python tools/check_docs.py            # markdown + the site
    .venv/bin/python tools/check_docs.py --quiet    # only failures

What it checks, across every git-tracked ``*.md`` plus ``docs/*.html``:

* every relative link and image target exists **and is tracked by git**;
* every ``#anchor`` resolves to a heading or an explicit ``id=`` in the target;
* every script named in a fenced shell block exists;
* every ``--flag`` used against a repo script appears in that script's argparse.

Exits non-zero if anything fails, so it can gate a commit.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LINK = re.compile(r'!?\[[^\]]*\]\(([^)\s]+)\)')
HTML_SRC = re.compile(r'(?:src|href)="([^"]+)"')
HEADING = re.compile(r'^#{1,6}\s+(.*)$', re.M)
EXPLICIT_ID = re.compile(r'id="([^"]+)"')
# a fenced block's python/py -m invocations, e.g. `python -m pursuit.sandbox` or
# `python tools/run_max.py --profile v1`
SCRIPT = re.compile(r'python3?\s+(?:-m\s+([\w.]+)|([\w./-]+\.py))')
FLAG = re.compile(r'\s(--[a-z][\w-]*)')


def tracked(rel: Path) -> bool:
    return subprocess.run(["git", "ls-files", "--error-unmatch", str(rel)],
                          cwd=ROOT, capture_output=True).returncode == 0


def slug(heading: str) -> str:
    """GitHub's anchor slug for a heading.

    Two details that are easy to get wrong and produce false failures:
    **each** whitespace character becomes its own hyphen (so ``a — b`` slugs to
    ``a--b``, not ``a-b``), and underscores survive while dots and slashes do
    not (``tools/run_max.py`` -> ``toolsrun_maxpy``).
    """
    h = re.sub(r'<[^>]+>', '', heading)                 # strip inline html
    h = re.sub(r'[`*]', '', h).strip().lower()          # markdown emphasis, not '_'
    h = re.sub(r'[^\w\s-]', '', h, flags=re.UNICODE)
    return re.sub(r'\s', '-', h)


def anchors(path: Path) -> set[str]:
    text = path.read_text(errors="replace")
    out = {slug(h) for h in HEADING.findall(text)}
    out |= set(EXPLICIT_ID.findall(text))
    return out


def argparse_flags(script: Path) -> set[str]:
    try:
        src = script.read_text(errors="replace")
    except OSError:
        return set()
    return set(re.findall(r'add_argument\(\s*"(--[\w-]+)"', src))


def check_targets(doc: Path, targets: list[str], fails: list, warns: list) -> None:
    for raw in targets:
        if raw.startswith(("http://", "https://", "mailto:", "data:", "#!")):
            continue
        path_part, _, frag = raw.partition("#")
        if not path_part:                                # in-page anchor
            if frag and frag not in anchors(doc):
                fails.append(f"{doc}: in-page anchor #{frag} has no heading")
            continue
        target = (doc.parent / path_part).resolve()
        try:
            rel = target.relative_to(ROOT)
        except ValueError:
            warns.append(f"{doc}: link escapes the repo -> {raw}")
            continue
        if not target.exists():
            fails.append(f"{doc}: MISSING -> {raw}")
            continue
        if target.is_file() and not tracked(rel):
            fails.append(f"{doc}: NOT TRACKED (404 on GitHub) -> {raw}")
            continue
        if frag and target.suffix == ".md" and frag not in anchors(target):
            fails.append(f"{doc}: anchor #{frag} not in {rel}")


def check_commands(doc: Path, fails: list, warns: list) -> None:
    text = doc.read_text(errors="replace")
    for block in re.findall(r'```(?:bash|sh|shell)?\n(.*?)```', text, re.S):
        for m in SCRIPT.finditer(block):
            module, script = m.group(1), m.group(2)
            line = block[m.start():block.find("\n", m.start()) % (len(block) + 1) or None]
            if script:
                p = ROOT / script
                if not p.is_file():
                    fails.append(f"{doc}: command names a missing script -> {script}")
                    continue
                known = argparse_flags(p)
            else:
                # `-m pkg` runs pkg/__main__.py; `-m pkg.mod` runs pkg/mod.py.
                # Anything with no file here is a third-party module (pytest, pip)
                # and not this repository's business.
                stem = ROOT / module.replace(".", "/")
                p = next((c for c in (stem.with_suffix(".py"), stem / "__main__.py")
                          if c.is_file()), None)
                if p is None:
                    if (ROOT / module.split(".")[0]).is_dir():
                        warns.append(f"{doc}: `-m {module}` has no module or __main__.py")
                    continue
                known = argparse_flags(p)
            if not known:
                continue
            for flag in FLAG.findall(line):
                if flag not in known:
                    warns.append(f"{doc}: `{flag}` not in {p.relative_to(ROOT)}'s argparse")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quiet", action="store_true", help="only print problems")
    a = ap.parse_args(argv)

    listed = subprocess.run(["git", "ls-files", "*.md", "docs/*.html"],
                            cwd=ROOT, capture_output=True, text=True).stdout.split()
    docs = [ROOT / f for f in listed]
    fails: list[str] = []
    warns: list[str] = []

    for doc in docs:
        text = doc.read_text(errors="replace")
        targets = LINK.findall(text) + HTML_SRC.findall(text)
        check_targets(doc, targets, fails, warns)
        if doc.suffix == ".md":
            check_commands(doc, fails, warns)

    if not a.quiet:
        print(f"checked {len(docs)} documents")
    for w in warns:
        print(f"  warn: {w}")
    for f in fails:
        print(f"  FAIL: {f}")
    if not fails and not a.quiet:
        print("  all links resolve and are tracked")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
