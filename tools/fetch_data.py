#!/usr/bin/env python3
"""Acquire the external datasets named in `benchmarks/catalog.py`.

The catalogue already knows what each dataset is, what it can prove, and **what stands
between you and the bytes** (`Dataset.gate`). This tool is the executable half of that:
it dispatches on the gate, and refuses to pretend.

Three decisions are load-bearing, and each came from a way this normally goes wrong:

1. **A gated dataset is not a download.** Five of the ten entries sit behind a web form,
   a signed agreement, BaiduYun, or no confirmed route at all. A fetcher that "tries
   anyway" writes a 4 KB HTML login page to `ARD100.zip`, extraction fails a week later,
   and the failure is blamed on the archive. Here those gates print exactly what the human
   must do -- including the address to email for the agreement -- and exit non-zero.
   Nothing is created on disk that could be mistaken for data.

2. **Never re-download 14 GB because a later step failed.** Presence, extraction and
   verification are separate and each is idempotent: an interrupted HTTP fetch resumes
   from `.part` with a Range header, an already-extracted tree is detected by its
   *layout* (not by a marker file we wrote, which a manual download would not have), and
   a manual drop of the archives into the dataset directory is picked up and extracted
   -- all of them, since a gated set is handed over in parts -- even when the gate says
   a human was required.

3. **Counting is verification; anything else is a guess.** After extraction the file
   counts are checked against the catalogue's `frames`/`sequences`, and a mismatch is
   printed loudly and exits non-zero rather than being folded into a summary line. Where
   no `Layout` is registered for a key, that is *said* -- an unverifiable tree is
   reported as unverifiable, never as verified.

ARD-MAV is a special case worth stating: it is already on disk, its directory is
`data/external/ard_mav/` while its catalogue key is `ardmav`, and re-fetching it would
cost 14.6 GB for nothing. The key -> directory mapping is explicit in `DIR_ALIASES`, and
an existing tree is never deleted.

    python tools/fetch_data.py --list
    python tools/fetch_data.py ardmav halmstad
    python tools/fetch_data.py --priority 2 --dry-run

Exit codes: 0 all good; 1 a fetch or a verification failed; 2 a human must act.
Stdlib only at import time (gdown is imported inside the function that needs it), so the
torch-free CI job can import and test this.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from benchmarks.catalog import DATASETS, Dataset, Gate, by_priority  # noqa: E402

DEFAULT_ROOT = REPO / "data" / "external"
MANIFEST_NAME = "MANIFEST.json"
# 2: `archive` (one record) became `archives` (a list) -- see manifest_dict.
MANIFEST_SCHEMA = 2

# Some datasets were unpacked before this tool existed and their directory name is not
# their catalogue key. Mapping the two by hand beats guessing: a wrong guess re-downloads
# 14.6 GB of ARD-MAV alongside the copy already on disk.
DIR_ALIASES: dict[str, str] = {"ardmav": "ard_mav"}

# The address to write to for an agreement-gated set. Kept as data rather than parsed out
# of `notes`, because a typo here costs a week of latency.
AGREEMENT_CONTACTS: dict[str, str] = {"dvb": "wosdetc@googlegroups.com"}

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")
_USER_AGENT = "SpeckLock-fetch_data/1 (+https://nadavcherry.github.io/SpeckLock/)"


# --------------------------------------------------------------------------- layouts

@dataclass(frozen=True)
class CountSpec:
    """One countable thing in an extracted tree, and the catalogue field it must match.

    `expect` names a `Dataset` attribute (`frames` or `sequences`) or is None when the
    count is informational only -- an informational count is reported but can never
    make the run claim the dataset was verified.
    """
    label: str
    glob: str
    expect: str | None = None


@dataclass(frozen=True)
class Layout:
    """How to recognise an already-extracted dataset and how to count it.

    `subdir` is the directory the archive unpacks into, relative to the dataset dir --
    empty when the archive has no top-level folder. Presence is decided by this layout
    and not by a marker file, because the datasets that arrive by hand (form, agreement,
    BaiduYun) never carry one.
    """
    subdir: str = ""
    counts: tuple[CountSpec, ...] = ()


# Registered only where the on-disk layout has actually been seen. An unregistered key
# still fetches and extracts; it is simply reported as unverifiable until someone adds
# its Layout here, which is the honest state of affairs.
LAYOUTS: dict[str, Layout] = {
    "ardmav": Layout(
        subdir="ARD-MAV",
        counts=(
            CountSpec("annotations", "Annotations/*/*.xml", "frames"),
            CountSpec("videos", "videos/*.mp4", "sequences"),
        ),
    ),
}

# Counted when no Layout is registered, so a fetched tree still gets a recorded shape.
GENERIC_COUNTS: tuple[CountSpec, ...] = (
    CountSpec("images", "**/*.jpg"),
    CountSpec("images_png", "**/*.png"),
    CountSpec("videos", "**/*.mp4"),
    CountSpec("annotations_xml", "**/*.xml"),
    CountSpec("annotations_txt", "**/*.txt"),
)


# --------------------------------------------------------------------------- selection

def dataset_dir(key: str, root: Path = DEFAULT_ROOT) -> Path:
    """Where a dataset lives on disk. See DIR_ALIASES for why this is not just the key."""
    return Path(root) / DIR_ALIASES.get(key, key)


def select(keys: list[str] | None = None, priority: int | None = None) -> list[Dataset]:
    """Resolve the CLI's keys and `--priority N` into datasets, in acquisition order.

    `--priority N` means "priority N or more urgent", i.e. the catalogue's *number* is
    at most N -- priority 1 is the most urgent, which is the opposite of how the flag
    reads out loud and so is stated here rather than left to the reader.
    """
    chosen: dict[str, Dataset] = {}
    for k in keys or []:
        if k not in DATASETS:
            raise KeyError(f"unknown dataset key {k!r}; known: {', '.join(sorted(DATASETS))}")
        chosen[k] = DATASETS[k]
    if priority is not None:
        for d in by_priority(priority):
            chosen[d.key] = d
    return sorted(chosen.values(), key=lambda d: (d.priority, d.key))


# --------------------------------------------------------------------------- presence

@dataclass
class Presence:
    """What is on disk for one dataset."""
    key: str
    directory: Path
    layout_root: Path
    present: bool
    counted: bool = False
    counts: dict[str, int] = field(default_factory=dict)
    archives: list[Path] = field(default_factory=list)
    has_manifest: bool = False

    @property
    def verifiable(self) -> bool:
        """False when no Layout is registered: the tree can be counted but the counts
        cannot be compared with the catalogue, so the run may not claim verification."""
        return self.key in LAYOUTS


def find_archives(directory: Path) -> list[Path]:
    """Archives sitting in the dataset directory, including ones a human dropped there.

    A partial download (`.part`) is deliberately not an archive: extracting a truncated
    zip produces a tree that looks plausible and is short a few thousand frames.
    """
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir()
                  if p.is_file() and any(p.name.endswith(s) for s in _ARCHIVE_SUFFIXES))


def count_tree(layout_root: Path, specs: tuple[CountSpec, ...]) -> dict[str, int]:
    if not layout_root.is_dir():
        return {}
    return {s.label: sum(1 for _ in layout_root.glob(s.glob)) for s in specs}


def probe(ds: Dataset, root: Path = DEFAULT_ROOT, *, deep: bool = False) -> Presence:
    """Is this dataset already on disk, and what is in it?

    Shallow by default (`--list` probes nine datasets and must stay instant); deep
    counting walks the tree, which on ARD-MAV is 107k files.
    """
    directory = dataset_dir(ds.key, root)
    layout = LAYOUTS.get(ds.key, Layout())
    layout_root = directory / layout.subdir if layout.subdir else directory

    present = False
    if layout_root.is_dir():
        present = any(p.name not in {MANIFEST_NAME} and not p.name.endswith(".part")
                      and not any(p.name.endswith(s) for s in _ARCHIVE_SUFFIXES)
                      for p in layout_root.iterdir())

    p = Presence(key=ds.key, directory=directory, layout_root=layout_root, present=present,
                 archives=find_archives(directory),
                 has_manifest=(directory / MANIFEST_NAME).is_file())
    if deep and present:
        specs = layout.counts or GENERIC_COUNTS
        # A registered layout keeps every count, including a zero -- a missing
        # Annotations/ is exactly the failure the count is there to catch. The generic
        # specs are guesses, so their empty rows are noise and are dropped.
        p.counts = {k: v for k, v in count_tree(layout_root, specs).items()
                    if v or layout.counts}
        p.counted = True
    return p


def check_counts(ds: Dataset, counts: dict[str, int], tolerance: float = 0.0) -> list[str]:
    """Compare counted files with the catalogue. Returns one string per mismatch.

    `tolerance` is a fraction and defaults to 0 for a reason: the counts that can be
    checked at all come from a registered Layout, and those were counted on the real
    tree, so any drift is a fact about the download rather than about rounding. Papers'
    rounded figures live on entries with no Layout, which are never checked here.
    """
    layout = LAYOUTS.get(ds.key)
    if layout is None:
        return []
    out: list[str] = []
    for spec in layout.counts:
        if spec.expect is None or spec.label not in counts:
            continue
        expected = getattr(ds, spec.expect, None)
        if not expected:
            continue
        got = counts[spec.label]
        if abs(got - expected) > tolerance * expected:
            out.append(f"{spec.label}: found {got:,}, catalogue {spec.expect}={expected:,} "
                       f"(delta {got - expected:+,})")
    return out


# --------------------------------------------------------------------------- gate plan

@dataclass(frozen=True)
class FetchPlan:
    """What to do about one dataset's gate. `blocked` means a human must act."""
    key: str
    action: str            # 'gdrive' | 'http' | 'manual'
    target: str = ""       # gdrive file id, or a direct URL
    message: str = ""
    blocked: bool = False


def _agreement_contact(ds: Dataset) -> str:
    if ds.key in AGREEMENT_CONTACTS:
        return AGREEMENT_CONTACTS[ds.key]
    m = _EMAIL_RE.search(ds.notes or "")
    return m.group(0) if m else "(no address recorded -- see the dataset page)"


def _manual_tail(ds: Dataset, root: Path = DEFAULT_ROOT) -> str:
    d = dataset_dir(ds.key, root)
    d = d.relative_to(REPO) if str(d).startswith(str(REPO)) else d
    return (f"  Then drop the archive in {d}/ and re-run:  "
            f"python tools/fetch_data.py {ds.key}\n"
            f"  (a local archive is extracted and verified even for a gated set)")


def plan_for(ds: Dataset, root: Path = DEFAULT_ROOT) -> FetchPlan:
    """Dispatch on the gate. The gated branches build instructions, never a request."""
    if ds.gate is Gate.GDRIVE:
        if ds.download_id:
            return FetchPlan(ds.key, "gdrive", ds.download_id,
                             f"Google Drive id {ds.download_id} (via gdown)")
        return FetchPlan(ds.key, "manual", blocked=True, message=(
            f"{ds.key}: gate=gdrive but no download_id in benchmarks/catalog.py.\n"
            f"  Open {ds.url}, find the Drive file id, and record it as download_id.\n"
            + _manual_tail(ds, root)))

    if ds.gate is Gate.OPEN:
        if ds.download_id.startswith(("http://", "https://")):
            return FetchPlan(ds.key, "http", ds.download_id, f"direct HTTP {ds.download_id}")
        return FetchPlan(ds.key, "manual", blocked=True, message=(
            f"{ds.key}: openly licensed, but no direct archive URL is recorded.\n"
            f"  Open {ds.url} and take the download link from the page.\n"
            f"  Record it as download_id=... in benchmarks/catalog.py to automate this.\n"
            + _manual_tail(ds, root)))

    if ds.gate is Gate.FORM:
        return FetchPlan(ds.key, "manual", blocked=True, message=(
            f"{ds.key}: access form. No automated route exists and none is attempted.\n"
            f"  1. Open {ds.url}\n"
            f"  2. Complete the access/registration form and wait for the link.\n"
            + _manual_tail(ds, root)))

    if ds.gate is Gate.AGREEMENT:
        return FetchPlan(ds.key, "manual", blocked=True, message=(
            f"{ds.key}: signed data-usage agreement. Budget a week of latency.\n"
            f"  1. Email {_agreement_contact(ds)} requesting the {ds.name} "
            f"data-usage agreement.\n"
            f"  2. Sign and return it; the download link comes back by email.\n"
            f"  Dataset page: {ds.url}\n"
            + _manual_tail(ds, root)))

    if ds.gate is Gate.BAIDU:
        return FetchPlan(ds.key, "manual", blocked=True, message=(
            f"{ds.key}: BaiduYun only, no mirror. Effectively manual outside China.\n"
            f"  1. Open {ds.url} and follow the BaiduYun (pan.baidu.com) link + extract code.\n"
            f"  2. A Baidu account is required; the web client throttles hard -- "
            f"expect hours.\n"
            + _manual_tail(ds, root)))

    return FetchPlan(ds.key, "manual", blocked=True, message=(
        f"{ds.key}: no verified download route "
        f"(catalogue verified={ds.verified}).\n"
        f"  Confirm the route by hand from {ds.url}, then record it in "
        f"benchmarks/catalog.py\n"
        f"  as download_id=... with the right Gate, so this stops being manual.\n"
        + _manual_tail(ds, root)))


# --------------------------------------------------------------------------- transfer

def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def archive_kind(path: Path) -> str | None:
    """'zip', 'tar', or None -- decided by content, not by the file name.

    gdown names a Drive download from the server's headers and an HTML error page can
    arrive as `dataset.zip`; sniffing is the only way to notice before extraction.
    """
    try:
        if zipfile.is_zipfile(path):
            return "zip"
        if tarfile.is_tarfile(path):
            return "tar"
    except (OSError, tarfile.TarError):
        return None
    return None


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def resume_mode(have: int, status: int) -> tuple[int, str]:
    """Given the bytes already on disk and the response status, where to start writing.

    Pulled out of `http_download` because it is the one piece of the transfer that can
    be wrong without failing: a server that ignores `Range` answers **200 with the
    entire body**, and appending that to a partial file yields an archive of plausible
    size that is corrupt in the middle. 206 is the only status that licenses an append.
    """
    if have and status == 206:
        return have, "ab"
    return 0, "wb"


def http_download(url: str, dest: Path, *, log=print, timeout: int = 60) -> Path:
    """Resumable HTTP GET. Bytes land in `<dest>.part` and are renamed only when done."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0

    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    if have:
        req.add_header("Range", f"bytes={have}-")
        log(f"    resuming at {_human(have)}")

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        start, mode = resume_mode(have, resp.status)
        if have and not start:
            log(f"    server ignored Range (HTTP {resp.status}); restarting from 0")
        have = start
        declared = int(resp.headers.get("Content-Length") or 0)
        total = declared + have
        got, t0 = have, time.time()
        with open(part, mode) as f:
            while True:
                block = resp.read(1 << 20)
                if not block:
                    break
                f.write(block)
                got += len(block)
                if total and time.time() - t0 > 2:
                    log(f"    {_human(got)} / {_human(total)}  ({100 * got / total:.1f} %)")
                    t0 = time.time()
    part.replace(dest)
    log(f"    downloaded {_human(dest.stat().st_size)} -> {dest.name}")
    return dest


def gdrive_download(file_id: str, dest: Path, *, log=print) -> Path:
    """Google Drive via gdown, which owns the confirm-token dance for large files.

    Imported here rather than at module scope so the torch-free CI job -- which installs
    neither gdown nor requests -- can still import and test this file.
    """
    try:
        import gdown  # noqa: PLC0415
    except ImportError as exc:                     # pragma: no cover - environment-dependent
        raise RuntimeError(
            "gdown is not installed; `.venv/bin/pip install gdown` or download "
            f"https://drive.google.com/file/d/{file_id}/view by hand") from exc
    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"    gdown id={file_id} -> {dest}")
    out = gdown.download(id=file_id, output=str(dest), quiet=False, resume=True)
    if out is None:
        raise RuntimeError(f"gdown returned nothing for id={file_id} "
                           "(quota exceeded, or the file is not shared)")
    return Path(out)


# --------------------------------------------------------------------------- extraction

def is_safe_member(name: str) -> bool:
    """Reject absolute paths and `..` escapes in an archive member.

    These archives come from third parties over links found in papers; one traversal
    member would write outside data/external/ with the repo's own permissions.
    """
    if not name:
        return False
    norm = name.replace("\\", "/")
    if norm.startswith("/") or re.match(r"^[A-Za-z]:", norm):
        return False
    return ".." not in norm.split("/")


def extract(archive: Path, into: Path, *, log=print) -> None:
    kind = archive_kind(archive)
    if kind is None:
        head = archive.read_bytes()[:64] if archive.stat().st_size else b""
        raise RuntimeError(
            f"{archive} is neither a zip nor a tar ({_human(archive.stat().st_size)}); "
            f"first bytes {head[:32]!r}. A login/error page, or a truncated download.")
    into.mkdir(parents=True, exist_ok=True)
    log(f"    extracting {kind} -> {into}")
    if kind == "zip":
        with zipfile.ZipFile(archive) as z:
            names = z.namelist()
            bad = [n for n in names if not is_safe_member(n)]
            if bad:
                raise RuntimeError(f"unsafe member(s) in {archive.name}: {bad[:3]}")
            z.extractall(into)
    else:
        # A name check is not enough for tar. A member `escape -> /tmp` followed by
        # `escape/x` writes outside `into` with every *name* looking harmless, and
        # `is_safe_member` only ever sees names. Only tarfile's own `data` filter
        # resolves link targets, so it does the work here and the name check stays as
        # the branch that produces a readable message. (zipfile does not restore
        # symlinks at all -- it writes the target as file content -- so the zip branch
        # is not exposed to this.) The filter argument landed in 3.11.4; the repo asks
        # for >=3.11, so on an older patch release we keep the old behaviour rather
        # than raising a TypeError after a 14 GB download.
        kw = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        with tarfile.open(archive) as t:
            members = t.getmembers()
            bad = [m.name for m in members if not is_safe_member(m.name)]
            if bad:
                raise RuntimeError(f"unsafe member(s) in {archive.name}: {bad[:3]}")
            t.extractall(into, **kw)


# --------------------------------------------------------------------------- manifest

def manifest_dict(ds: Dataset, presence: Presence, *,
                  archives: Sequence[Path] = (),
                  sha256: dict[str, str] | None = None,
                  source_url: str = "",
                  mismatches: list[str] | None = None,
                  note: str = "") -> dict:
    """The record left beside the data: where it came from and what was checked.

    `catalogue_verified` is copied in deliberately. It is the catalogue's claim that the
    *route* was confirmed by a human, which is a different assertion from "the files on
    this disk match the expected counts" (`counts_verified`), and conflating the two is
    how an unverified route ends up cited as evidence.

    `archives` is a list, not one archive, because a gated dataset does not arrive as a
    single file -- Halmstad ships IR and visible separately, and the agreement sets come
    in parts. A record naming only the first is a record that says a fetch completed
    when it half-did.
    """
    def rel(p: Path) -> str:
        try:
            return str(Path(p).resolve().relative_to(REPO))
        except ValueError:
            return str(p)

    expected = {}
    layout = LAYOUTS.get(ds.key)
    if layout:
        for spec in layout.counts:
            if spec.expect and getattr(ds, spec.expect, None):
                expected[spec.label] = getattr(ds, spec.expect)

    shas = sha256 or {}
    return {
        "schema": MANIFEST_SCHEMA,
        "tool": "tools/fetch_data.py",
        "key": ds.key,
        "name": ds.name,
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gate": ds.gate.value,
        "source_url": source_url or ds.url,
        "download_id": ds.download_id,
        "licence": ds.licence,
        "directory": rel(presence.directory),
        "layout_root": rel(presence.layout_root),
        "archives": [{
            "name": a.name,
            "bytes": a.stat().st_size if a.exists() else None,
            "sha256": shas.get(a.name),
        } for a in archives],
        "counts": dict(presence.counts),
        "expected": expected,
        "mismatches": list(mismatches or []),
        "counts_verified": bool(presence.counted and presence.verifiable
                                and expected and not mismatches),
        "catalogue_verified": ds.verified,
        "note": note,
    }


def write_manifest(payload: dict, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MANIFEST_NAME
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


# --------------------------------------------------------------------------- listing

def _conditions(ds: Dataset, width: int = 34) -> str:
    s = ",".join(c.value for c in ds.conditions)
    return s if len(s) <= width else s[:width - 1] + "…"


def print_table(datasets: list[Dataset], root: Path) -> None:
    hdr = (f"{'key':<14} {'pri':>3} {'gate':<9} {'size':>7} {'licence':<24} "
           f"{'conditions':<34} {'on disk':<9}")
    print(hdr)
    print("-" * len(hdr))
    for ds in datasets:
        p = probe(ds, root)
        if p.present:
            state = "yes"
        elif p.archives:
            state = "archive"
        else:
            state = "-"
        size = f"{ds.size_gb:.1f} GB" if ds.size_gb else "?"
        lic = ds.licence if len(ds.licence) <= 24 else ds.licence[:23] + "…"
        print(f"{ds.key:<14} {ds.priority:>3} {ds.gate.value:<9} {size:>7} {lic:<24} "
              f"{_conditions(ds):<34} {state:<9}")
    print(f"\n{len(datasets)} dataset(s); root {root}")
    print("'archive' = an archive is present but nothing is extracted yet; re-run the key "
          "to extract it.")


# --------------------------------------------------------------------------- one dataset

def _archive_path(ds: Dataset, directory: Path, plan: FetchPlan) -> Path:
    """Where a downloaded archive lands. Named from the URL when there is one, so a
    resumed run finds the same `.part`; the suffix is only a hint -- `archive_kind`
    decides what it really is."""
    if plan.action == "http":
        name = plan.target.rstrip("/").rsplit("/", 1)[-1].split("?")[0]
        if any(name.endswith(s) for s in _ARCHIVE_SUFFIXES):
            return directory / name
    return directory / f"{ds.key}.zip"


def fetch_one(ds: Dataset, root: Path, args, log=print) -> str:
    """Fetch/verify one dataset. Returns 'ok' | 'skipped' | 'blocked' | 'failed'."""
    directory = dataset_dir(ds.key, root)
    log(f"\n=== {ds.key}  ({ds.name}, priority {ds.priority}, gate {ds.gate.value})")
    log(f"    dir {directory}")

    p = probe(ds, root, deep=not args.quick)
    archives: list[Path] = []
    shas: dict[str, str] = {}
    source_url = ""
    note = ""

    if p.present and not args.redownload:
        log(f"    already extracted at {p.layout_root} -- not re-downloading")
        if args.dry_run:
            return "skipped"
    else:
        local = p.archives
        plan = plan_for(ds, root)
        if local and not args.redownload:
            # Every archive here, not just the first. A form/agreement/BaiduYun dataset
            # is handed over as several files, and extracting only `local[0]` leaves a
            # tree that looks plausible, reports 'ok', and is then skipped as 'present'
            # on every later run -- the rest of the dataset silently never arrives.
            archives = list(local)
            log(f"    found {len(archives)} local archive(s) -- extracting, not downloading: "
                + ", ".join(f"{a.name} ({_human(a.stat().st_size)})" for a in archives))
            note = "extracted from archive(s) already present on disk"
        elif plan.blocked:
            log("    HUMAN ACTION REQUIRED -- nothing was downloaded:\n")
            log(plan.message)
            return "blocked"
        else:
            dest = _archive_path(ds, directory, plan)
            source_url = plan.target if plan.action == "http" else \
                f"https://drive.google.com/file/d/{plan.target}/view"
            if args.dry_run:
                log(f"    DRY RUN: would {plan.action} {plan.message} -> {dest}")
                return "skipped"
            try:
                if plan.action == "gdrive":
                    archives = [gdrive_download(plan.target, dest, log=log)]
                else:
                    archives = [http_download(plan.target, dest, log=log)]
            except (OSError, urllib.error.URLError, RuntimeError) as exc:
                log(f"    FAILED to download: {exc}")
                return "failed"

        if args.dry_run:
            log(f"    DRY RUN: would extract {', '.join(a.name for a in archives)}")
            return "skipped"
        try:
            for a in archives:
                shas[a.name] = sha256_file(a)
                log(f"    sha256 {a.name} {shas[a.name]}")
                extract(a, directory, log=log)
        except (OSError, RuntimeError, zipfile.BadZipFile, tarfile.TarError) as exc:
            log(f"    FAILED to extract: {exc}")
            return "failed"
        p = probe(ds, root, deep=not args.quick)
        if not p.present:
            log(f"    FAILED: nothing recognisable under {p.layout_root} after extraction")
            return "failed"

    if not archives and p.archives:
        # Present already: the archives are recorded but nothing is re-extracted over an
        # existing tree, and hashing is opt-in -- sha256 of the 14.6 GB ARD-MAV zip takes
        # minutes and would run on every `--list`-adjacent invocation otherwise.
        archives = list(p.archives)
        if args.hash:
            for a in archives:
                shas[a.name] = sha256_file(a)
                log(f"    sha256 {a.name} {shas[a.name]}")
        else:
            note = (note or "") + ("; " if note else "") + \
                "sha256 of the pre-existing archive(s) not computed (pass --hash)"

    status = "ok"
    mismatches: list[str] = []
    if p.counted:
        for label, n in p.counts.items():
            log(f"    {label:<18} {n:>9,}")
        mismatches = check_counts(ds, p.counts, args.tolerance)
        if mismatches:
            log("    !! COUNT MISMATCH -- do not use this copy until it is explained:")
            for m in mismatches:
                log(f"       {m}")
            status = "failed"
        elif not p.verifiable:
            log(f"    NOT VERIFIED: no Layout registered for {ds.key} in tools/fetch_data.py, "
                "so these counts cannot be checked against the catalogue.")
        else:
            log("    counts match the catalogue")
    else:
        log("    counts not taken (--quick)")

    if not args.quick:
        mp = write_manifest(manifest_dict(ds, p, archives=archives, sha256=shas,
                                          source_url=source_url, mismatches=mismatches,
                                          note=note), directory)
        log(f"    manifest {mp}")
    return status


# --------------------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("keys", nargs="*", help="catalogue keys (see --list)")
    ap.add_argument("--list", action="store_true", help="print the catalogue and exit")
    ap.add_argument("--priority", type=int, default=None,
                    help="also fetch every dataset at priority N or more urgent (N or lower)")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan; touch nothing")
    ap.add_argument("--redownload", action="store_true",
                    help="fetch again even if a copy is present (extracts OVER it in place)")
    ap.add_argument("--hash", action="store_true",
                    help="sha256 an archive that was already on disk (minutes on 14 GB)")
    ap.add_argument("--quick", action="store_true",
                    help="skip file counting and the manifest")
    ap.add_argument("--tolerance", type=float, default=0.0,
                    help="allowed fractional deviation from the catalogue counts")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root)

    if args.list:
        print_table(sorted(DATASETS.values(), key=lambda d: (d.priority, d.key)), root)
        return 0

    if not args.keys and args.priority is None:
        build_parser().print_usage()
        print("\nNothing selected. Pass one or more keys, or --priority N, or --list.")
        return 2

    try:
        datasets = select(args.keys, args.priority)
    except KeyError as exc:
        print(exc.args[0])
        return 2

    results = {ds.key: fetch_one(ds, root, args) for ds in datasets}

    print("\n" + "=" * 72)
    for key, status in results.items():
        print(f"  {key:<14} {status}")
    blocked = [k for k, s in results.items() if s == "blocked"]
    failed = [k for k, s in results.items() if s == "failed"]
    if failed:
        print(f"\nFAILED: {', '.join(failed)}")
        return 1
    if blocked:
        print(f"\nNeeds a human: {', '.join(blocked)} -- see the instructions above.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
