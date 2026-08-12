"""Tests for the dataset fetcher's decisions -- never for its transfers.

Everything here runs offline against a faked `data/external` under `tmp_path`. What is
worth testing is the part that is *judgement* rather than I/O:

* a gated dataset must never be treated as a download (five of the ten entries are
  gated, and the failure mode is a 4 KB HTML login page saved as `ARD100.zip`);
* an already-extracted tree must be recognised **by its layout**, because ARD-MAV was
  unpacked long before this tool existed, lives under `ard_mav/` rather than its
  catalogue key `ardmav`, and re-fetching it costs 14.6 GB;
* a count that disagrees with the catalogue must come back as a mismatch, and a tree
  with no registered layout must come back as *unverifiable* rather than as verified.

The module is stdlib-only by design, so these tests need neither torch, ultralytics,
gdown nor a network.
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import fetch_data as F  # noqa: E402

from benchmarks.catalog import DATASETS, Condition, Dataset, Gate  # noqa: E402


# --------------------------------------------------------------------------- helpers

def make_ardmav(root: Path, *, n_xml: int = 3, n_videos: int = 2) -> Path:
    """The real ARD-MAV layout in miniature: dir 'ard_mav', subdir 'ARD-MAV'."""
    base = root / "ard_mav" / "ARD-MAV"
    (base / "Annotations" / "phantom02").mkdir(parents=True)
    (base / "videos").mkdir(parents=True)
    for i in range(n_xml):
        (base / "Annotations" / "phantom02" / f"{i:06d}.xml").write_text("<annotation/>", encoding="utf-8")
    for i in range(n_videos):
        (base / "videos" / f"phantom{i:02d}.mp4").write_bytes(b"\x00")
    return base


def tiny_ardmav(**kw) -> Dataset:
    """The catalogue entry with counts small enough to assert against."""
    from dataclasses import replace
    return replace(DATASETS["ardmav"], **kw)


class Args:
    """Stand-in for the argparse namespace `fetch_one` reads."""
    def __init__(self, **kw):
        self.dry_run = kw.get("dry_run", True)
        self.redownload = kw.get("redownload", False)
        self.hash = kw.get("hash", False)
        self.quick = kw.get("quick", False)
        self.tolerance = kw.get("tolerance", 0.0)


# --------------------------------------------------------------------------- gates

def test_gdrive_dispatches_to_gdown_with_the_catalogue_id():
    plan = F.plan_for(DATASETS["ardmav"])
    assert plan.action == "gdrive" and not plan.blocked
    assert plan.target == DATASETS["ardmav"].download_id


def test_open_with_a_direct_url_dispatches_to_http():
    ds = tiny_ardmav(gate=Gate.OPEN, download_id="https://example.org/x.zip")
    plan = F.plan_for(ds)
    assert plan.action == "http" and not plan.blocked
    assert plan.target == "https://example.org/x.zip"


def test_open_without_a_url_is_manual_and_not_a_pretend_download():
    """Four OPEN entries have no archive URL recorded. Guessing one is how a login page
    gets saved as the dataset."""
    ds = DATASETS["halmstad"]
    assert ds.gate is Gate.OPEN and not ds.download_id
    plan = F.plan_for(ds)
    assert plan.action == "manual" and plan.blocked
    assert ds.url in plan.message


@pytest.mark.parametrize("gate", [Gate.FORM, Gate.AGREEMENT, Gate.BAIDU, Gate.UNKNOWN])
def test_every_human_gate_is_blocked_and_says_what_to_do(gate):
    ds = tiny_ardmav(gate=gate, download_id="")
    plan = F.plan_for(ds)
    assert plan.blocked and plan.action == "manual"
    assert plan.message.strip()
    assert ds.url in plan.message
    assert "fetch_data.py ardmav" in plan.message      # the resume instruction


def test_agreement_gate_prints_the_email_address():
    plan = F.plan_for(DATASETS["dvb"])
    assert plan.blocked
    assert "wosdetc@googlegroups.com" in plan.message


def test_gdrive_without_an_id_does_not_become_a_download():
    plan = F.plan_for(tiny_ardmav(download_id=""))
    assert plan.blocked and plan.action == "manual"


def test_every_catalogue_entry_has_a_plan():
    for ds in DATASETS.values():
        plan = F.plan_for(ds)
        assert plan.action in {"gdrive", "http", "manual"}
        assert plan.blocked == (plan.action == "manual")


# --------------------------------------------------------------------------- presence

def test_ardmav_directory_alias_is_explicit(tmp_path):
    assert F.dataset_dir("ardmav", tmp_path) == tmp_path / "ard_mav"
    assert F.dataset_dir("halmstad", tmp_path) == tmp_path / "halmstad"


def test_existing_ardmav_tree_is_detected_as_present(tmp_path):
    make_ardmav(tmp_path)
    p = F.probe(DATASETS["ardmav"], tmp_path, deep=True)
    assert p.present and p.verifiable
    assert p.layout_root == tmp_path / "ard_mav" / "ARD-MAV"
    assert p.counts == {"annotations": 3, "videos": 2}


def test_missing_dataset_is_absent(tmp_path):
    p = F.probe(DATASETS["ardmav"], tmp_path, deep=True)
    assert not p.present and p.counts == {}


def test_a_directory_holding_only_the_archive_is_not_present(tmp_path):
    """An interrupted run leaves the zip and no tree. Calling that 'present' would skip
    the extraction forever."""
    d = tmp_path / "halmstad"
    d.mkdir()
    (d / "halmstad.zip").write_bytes(b"PK")
    (d / "halmstad.zip.part").write_bytes(b"PK")
    p = F.probe(DATASETS["halmstad"], tmp_path, deep=True)
    assert not p.present
    assert [a.name for a in p.archives] == ["halmstad.zip"]   # the .part is not an archive


def test_unregistered_layout_is_reported_unverifiable_but_still_counted(tmp_path):
    d = tmp_path / "smot4sb" / "images"
    d.mkdir(parents=True)
    (d / "a.jpg").write_bytes(b"\xff\xd8")
    (d / "b.jpg").write_bytes(b"\xff\xd8")
    p = F.probe(DATASETS["smot4sb"], tmp_path, deep=True)
    assert p.present and not p.verifiable
    assert p.counts["images"] == 2
    assert "videos" not in p.counts                 # empty generic rows are dropped


def test_shallow_probe_does_not_count(tmp_path):
    make_ardmav(tmp_path)
    p = F.probe(DATASETS["ardmav"], tmp_path)
    assert p.present and not p.counted and p.counts == {}


# --------------------------------------------------------------------------- counts

def test_count_mismatch_is_reported():
    ds = tiny_ardmav(frames=10, sequences=2)
    bad = F.check_counts(ds, {"annotations": 7, "videos": 2})
    assert len(bad) == 1
    assert "found 7" in bad[0] and "-3" in bad[0]


def test_counts_that_match_produce_no_complaint():
    ds = tiny_ardmav(frames=10, sequences=2)
    assert F.check_counts(ds, {"annotations": 10, "videos": 2}) == []


def test_tolerance_is_fractional_and_off_by_default():
    ds = tiny_ardmav(frames=1000, sequences=2)
    assert F.check_counts(ds, {"annotations": 995, "videos": 2}) != []
    assert F.check_counts(ds, {"annotations": 995, "videos": 2}, tolerance=0.01) == []


def test_unregistered_layout_is_never_silently_verified():
    """No Layout means no claim: `check_counts` must not invent a comparison."""
    assert F.check_counts(DATASETS["smot4sb"], {"images": 5}) == []


def test_real_ardmav_catalogue_counts_are_the_ones_the_layout_checks():
    layout = F.LAYOUTS["ardmav"]
    assert {c.expect for c in layout.counts} == {"frames", "sequences"}
    assert DATASETS["ardmav"].frames == 107497 and DATASETS["ardmav"].sequences == 60


# --------------------------------------------------------------------------- manifest

def test_manifest_shape_and_verified_flags(tmp_path):
    make_ardmav(tmp_path)
    ds = tiny_ardmav(frames=3, sequences=2)
    p = F.probe(ds, tmp_path, deep=True)
    archive = tmp_path / "ard_mav" / "ARD-MAV.zip"
    archive.write_bytes(b"PK\x03\x04")
    payload = F.manifest_dict(ds, p, archives=[archive], sha256={"ARD-MAV.zip": "deadbeef"},
                              source_url="https://example.org/a.zip")
    F.write_manifest(payload, p.directory)

    got = json.loads((p.directory / "MANIFEST.json").read_text(encoding="utf-8"))
    for key in ("schema", "key", "fetched_at", "source_url", "archives", "counts",
                "expected", "mismatches", "counts_verified", "catalogue_verified"):
        assert key in got, key
    assert got["key"] == "ardmav"
    assert got["archives"][0]["sha256"] == "deadbeef"
    assert got["archives"][0]["bytes"] == 4
    assert got["counts"] == {"annotations": 3, "videos": 2}
    assert got["expected"] == {"annotations": 3, "videos": 2}
    assert got["counts_verified"] is True
    assert got["catalogue_verified"] is DATASETS["ardmav"].verified
    assert got["fetched_at"].endswith("Z")


def test_manifest_separates_route_verification_from_count_verification(tmp_path):
    """`catalogue_verified` says a human confirmed the download route; `counts_verified`
    says the bytes on this disk match. Conflating them is how an unverified dataset gets
    cited as evidence."""
    d = tmp_path / "smot4sb" / "images"
    d.mkdir(parents=True)
    (d / "a.jpg").write_bytes(b"\xff\xd8")
    ds = DATASETS["smot4sb"]
    p = F.probe(ds, tmp_path, deep=True)
    m = F.manifest_dict(ds, p)
    assert m["catalogue_verified"] is True          # the route was confirmed
    assert m["counts_verified"] is False            # no layout, so no count claim
    assert m["expected"] == {}


def test_manifest_records_mismatches_rather_than_dropping_them(tmp_path):
    make_ardmav(tmp_path, n_xml=1)
    ds = tiny_ardmav(frames=99, sequences=2)
    p = F.probe(ds, tmp_path, deep=True)
    bad = F.check_counts(ds, p.counts)
    m = F.manifest_dict(ds, p, mismatches=bad)
    assert m["mismatches"] and m["counts_verified"] is False


# --------------------------------------------------------------------------- selection

def test_priority_filter_takes_everything_at_or_above_the_level():
    keys = {d.key for d in F.select(priority=2)}
    assert "ardmav" in keys and "halmstad" in keys and "uav_smid" in keys
    assert "ard100" not in keys and "dut_antiuav" not in keys
    assert all(DATASETS[k].priority <= 2 for k in keys)


def test_priority_and_explicit_keys_union_without_duplicates():
    ds = F.select(["ard100", "ardmav"], priority=1)
    keys = [d.key for d in ds]
    assert keys == sorted(set(keys), key=lambda k: (DATASETS[k].priority, k))
    assert set(keys) == {"ardmav", "ard100"}


def test_selection_is_in_acquisition_order():
    ds = F.select(priority=99)
    assert [d.priority for d in ds] == sorted(d.priority for d in ds)


def test_unknown_key_is_rejected_with_the_valid_ones():
    with pytest.raises(KeyError) as e:
        F.select(["ard_mav"])                       # the directory name, not the key
    assert "ardmav" in e.value.args[0]


# --------------------------------------------------------------------------- archives

def test_archive_kind_sniffs_content_not_the_name(tmp_path):
    z = tmp_path / "looks_like_data.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("a.txt", "x")
    assert F.archive_kind(z) == "zip"

    html = tmp_path / "also.zip"
    html.write_text("<!DOCTYPE html><title>Sign in</title>", encoding="utf-8")
    assert F.archive_kind(html) is None


def test_extract_refuses_a_traversal_member(tmp_path):
    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("../escaped.txt", "x")
    with pytest.raises(RuntimeError, match="unsafe member"):
        F.extract(z, tmp_path / "out")
    assert not (tmp_path / "escaped.txt").exists()


@pytest.mark.parametrize("name,ok", [
    ("ARD-MAV/videos/a.mp4", True),
    ("a.txt", True),
    ("../a.txt", False),
    ("/etc/passwd", False),
    ("C:\\windows\\x", False),
    ("dir/../../x", False),
    ("", False),
])
def test_is_safe_member(name, ok):
    assert F.is_safe_member(name) is ok


def test_extract_refuses_a_tar_symlink_escape(tmp_path):
    """The traversal check reads member *names*, and these two names are both harmless:
    a symlink `escape -> <outside>` followed by a regular member `escape/pwned.txt`.
    Only tarfile's `data` filter resolves the link target. Without it this wrote a file
    outside the destination on this machine."""
    import io
    import tarfile

    outside = tmp_path / "outside"
    outside.mkdir()
    archive = tmp_path / "evil.tar"
    with tarfile.open(archive, "w") as t:
        link = tarfile.TarInfo("escape")
        link.type = tarfile.SYMTYPE
        link.linkname = str(outside)
        t.addfile(link)
        payload = b"pwned"
        member = tarfile.TarInfo("escape/pwned.txt")
        member.size = len(payload)
        t.addfile(member, io.BytesIO(payload))

    with pytest.raises((RuntimeError, tarfile.TarError)):
        F.extract(archive, tmp_path / "out")
    assert not (outside / "pwned.txt").exists()


def test_extract_rejects_a_non_archive_with_a_readable_message(tmp_path):
    p = tmp_path / "x.zip"
    p.write_text("<html>login</html>", encoding="utf-8")
    with pytest.raises(RuntimeError, match="neither a zip nor a tar"):
        F.extract(p, tmp_path / "out")


@pytest.mark.parametrize("have,status,expect", [
    (0, 200, (0, "wb")),
    (0, 206, (0, "wb")),
    (1024, 206, (1024, "ab")),          # the only case that may append
    (1024, 200, (0, "wb")),             # Range ignored -> the body starts at byte 0
    (1024, 416, (0, "wb")),
])
def test_resume_mode_only_appends_on_206(have, status, expect):
    """Appending a 200 body onto a partial file gives an archive of the right size that
    is corrupt in the middle -- the failure this branch exists to prevent."""
    assert F.resume_mode(have, status) == expect


class FakeResponse:
    """The two attributes `http_download` reads off urlopen, and nothing else."""
    def __init__(self, body: bytes, status: int):
        self._b = io.BytesIO(body)
        self.status = status
        self.headers = {"Content-Length": str(len(body))}

    def read(self, n: int) -> bytes:
        return self._b.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.mark.parametrize("status", [206, 200])
def test_http_download_reassembles_the_right_bytes_whether_or_not_range_is_honoured(
        tmp_path, monkeypatch, status):
    """`resume_mode` is unit-tested, but the bug it prevents only shows up once the
    decision is applied to a file on disk: a server answering 200 to a Range request
    sends the WHOLE body, and appending it to `.part` yields an archive of plausible
    size that is corrupt in the middle."""
    body = b"A" * 100 + b"B" * 100
    dest = tmp_path / "x.zip"
    dest.with_name("x.zip.part").write_bytes(body[:100])

    def urlopen(req, timeout=None):
        assert req.get_header("Range") == "bytes=100-"
        return FakeResponse(body[100:] if status == 206 else body, status)

    monkeypatch.setattr(F.urllib.request, "urlopen", urlopen)
    F.http_download("https://example.org/x.zip", dest, log=lambda s: None)

    assert dest.read_bytes() == body
    assert not dest.with_name("x.zip.part").exists()   # renamed, not left behind


def test_sha256_matches_hashlib(tmp_path):
    import hashlib
    p = tmp_path / "b.bin"
    p.write_bytes(b"speck" * 1000)
    assert F.sha256_file(p) == hashlib.sha256(b"speck" * 1000).hexdigest()


# --------------------------------------------------------------------------- flow

def test_present_dataset_is_skipped_and_never_downloaded(tmp_path, monkeypatch):
    """The ARD-MAV case: 14.6 GB already on disk must not be fetched again."""
    make_ardmav(tmp_path)

    def boom(*a, **k):                              # noqa: ANN002
        raise AssertionError("a download was attempted for a dataset already on disk")
    monkeypatch.setattr(F, "gdrive_download", boom)
    monkeypatch.setattr(F, "http_download", boom)

    ds = tiny_ardmav(frames=3, sequences=2)
    lines: list[str] = []
    status = F.fetch_one(ds, tmp_path, Args(dry_run=False), log=lines.append)
    assert status == "ok"
    assert any("already extracted" in ln for ln in lines)
    assert (tmp_path / "ard_mav" / "MANIFEST.json").is_file()


def test_present_dataset_with_wrong_counts_fails_loudly(tmp_path, monkeypatch):
    make_ardmav(tmp_path, n_xml=1)
    monkeypatch.setattr(F, "gdrive_download", lambda *a, **k: pytest.fail("downloaded"))
    ds = tiny_ardmav(frames=500, sequences=2)
    lines: list[str] = []
    assert F.fetch_one(ds, tmp_path, Args(dry_run=False), log=lines.append) == "failed"
    assert any("COUNT MISMATCH" in ln for ln in lines)
    got = json.loads((tmp_path / "ard_mav" / "MANIFEST.json").read_text(encoding="utf-8"))
    assert got["mismatches"] and got["counts_verified"] is False


def test_blocked_gate_creates_nothing_on_disk(tmp_path):
    lines: list[str] = []
    status = F.fetch_one(DATASETS["dvb"], tmp_path, Args(dry_run=False), log=lines.append)
    assert status == "blocked"
    assert not (tmp_path / "dvb").exists()
    assert any("HUMAN ACTION REQUIRED" in ln for ln in lines)


def test_a_manually_dropped_archive_is_extracted_even_for_a_gated_set(tmp_path):
    """The whole point of the gated branch: the human does the download, the tool still
    does the extraction, the counting and the manifest."""
    d = tmp_path / "dvb"
    d.mkdir()
    with zipfile.ZipFile(d / "dvb.zip", "w") as zf:
        zf.writestr("seq01/000001.jpg", "x")
    lines: list[str] = []
    status = F.fetch_one(DATASETS["dvb"], tmp_path, Args(dry_run=False), log=lines.append)
    assert status == "ok"
    assert (d / "seq01" / "000001.jpg").is_file()
    got = json.loads((d / "MANIFEST.json").read_text(encoding="utf-8"))
    assert [a["name"] for a in got["archives"]] == ["dvb.zip"]
    assert got["archives"][0]["sha256"]
    assert got["counts_verified"] is False          # no Layout registered for dvb


def test_every_dropped_archive_is_extracted_not_just_the_first(tmp_path):
    """A gated set is handed over in parts -- Halmstad ships IR and visible separately.
    Extracting only `archives[0]` leaves a tree that looks plausible, reports 'ok', and
    is skipped as 'present' forever after, so the rest of the dataset silently never
    arrives and no count can catch it (these are the keys with no Layout)."""
    d = tmp_path / "halmstad"
    d.mkdir()
    for i in (1, 2, 3):
        with zipfile.ZipFile(d / f"part{i}.zip", "w") as zf:
            zf.writestr(f"visible/vid{i}.mp4", "x")

    status = F.fetch_one(DATASETS["halmstad"], tmp_path, Args(dry_run=False), log=lambda s: None)
    assert status == "ok"
    assert sorted(p.name for p in (d / "visible").iterdir()) == ["vid1.mp4", "vid2.mp4", "vid3.mp4"]

    got = json.loads((d / "MANIFEST.json").read_text(encoding="utf-8"))
    assert [a["name"] for a in got["archives"]] == ["part1.zip", "part2.zip", "part3.zip"]
    assert all(a["sha256"] for a in got["archives"])


def test_dry_run_touches_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "gdrive_download", lambda *a, **k: pytest.fail("downloaded"))
    status = F.fetch_one(DATASETS["ardmav"], tmp_path, Args(dry_run=True), log=lambda s: None)
    assert status == "skipped"
    assert not (tmp_path / "ard_mav").exists()


def test_cli_list_runs_offline(tmp_path, capsys):
    assert F.main(["--list", "--root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "ardmav" in out and "gdrive" in out and "agreement" in out


def test_cli_returns_2_when_nothing_is_selected(capsys):
    assert F.main([]) == 2


def test_cli_returns_2_for_a_gated_dataset(tmp_path, capsys):
    assert F.main(["dvb", "--root", str(tmp_path)]) == 2
    assert "wosdetc@googlegroups.com" in capsys.readouterr().out


def test_cli_returns_2_for_an_unknown_key(tmp_path, capsys):
    assert F.main(["nope", "--root", str(tmp_path)]) == 2


def test_module_imports_without_torch_or_gdown():
    """CI installs numpy/scipy/opencv/pytest only. A module-scope `import gdown` here
    would pass locally and fail the whole test job.

    Parsed rather than grepped: reading the source text down to `def gdrive_download`
    and searching that prefix is blind to an import added anywhere below it, which is
    most of the file.
    """
    import ast

    banned = {"gdown", "torch", "torchvision", "ultralytics", "requests", "yaml", "pandas"}
    tree = ast.parse((REPO / "tools" / "fetch_data.py").read_text(encoding="utf-8"))
    in_function = {node for fn in ast.walk(tree)
                   if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
                   for node in ast.walk(fn)}

    for node in ast.walk(tree):
        if node in in_function or not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        modules = ([a.name for a in node.names] if isinstance(node, ast.Import)
                   else [node.module or ""])
        for name in modules:
            assert name.split(".")[0] not in banned, \
                f"line {node.lineno}: {name} must be imported inside the function that needs it"


def test_a_checksum_mismatch_refuses_to_extract(tmp_path):
    """A truncated 3 GB download extracts without complaint and yields a dataset quietly
    missing its tail, which surfaces later as a mysteriously poor model rather than as a
    failed fetch. Where the host publishes a checksum, a mismatch must stop the run."""
    from dataclasses import replace
    d = tmp_path / "uav_smid"
    d.mkdir()
    with zipfile.ZipFile(d / "UAV_SMID.zip", "w") as zf:
        zf.writestr("images/a.jpg", "x")
    ds = replace(DATASETS["uav_smid"], sha256="0" * 64)     # cannot match real content

    lines: list[str] = []
    status = F.fetch_one(ds, tmp_path, Args(dry_run=False), log=lines.append)

    assert status == "failed"
    assert any("checksum mismatch" in ln for ln in lines)
    assert not (d / "images").exists(), "nothing may be extracted after a mismatch"


def test_a_matching_checksum_extracts_normally(tmp_path):
    from dataclasses import replace
    d = tmp_path / "uav_smid"
    d.mkdir()
    archive = d / "UAV_SMID.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("images/a.jpg", "x")
    ds = replace(DATASETS["uav_smid"], sha256=F.sha256_file(archive))

    status = F.fetch_one(ds, tmp_path, Args(dry_run=False), log=lambda s: None)
    assert status == "ok"
    assert (d / "images" / "a.jpg").is_file()


def test_no_recorded_checksum_means_no_check(tmp_path):
    """Most hosts publish nothing; absence must not block the fetch."""
    from dataclasses import replace
    d = tmp_path / "uav_smid"
    d.mkdir()
    with zipfile.ZipFile(d / "UAV_SMID.zip", "w") as zf:
        zf.writestr("images/a.jpg", "x")
    ds = replace(DATASETS["uav_smid"], sha256="")

    assert F.fetch_one(ds, tmp_path, Args(dry_run=False), log=lambda s: None) == "ok"
