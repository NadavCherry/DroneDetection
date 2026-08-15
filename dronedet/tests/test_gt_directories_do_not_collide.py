"""The GT directory IS the test set, so two builders must never share one.

`tools/evaluate.py` defines the evaluated sequences as `gt_dir.glob("*.json")`. There is no
manifest and no split file: whatever JSONs are in the directory are what gets scored. That
makes the output path of every GT builder a correctness-critical choice rather than a
naming preference.

Two NPS builders used to write to `work/ext_datasets/gt/nps`:

    build_nps_test_gt()           all 50 clips, Purdue v1 annotations
    build_nps_test_gt_dogfight()  the 10 test clips, Dogfight annotations

Running the first would have made the "test set" 50 clips, 36 of them TRAINING clips, and
reported an AP over data the models were fitted on. Worse, whichever ran last silently
decided the corner convention for the 10 real test clips -- Purdue v1 is (y1,x1,y2,x2),
Dogfight is (x1,y1,x2,y2) -- and a transposed corner on a 3-14 px target is a total miss
that shows up as a merely disappointing score.

Every published NPS number we compare against is on Dogfight's re-annotations, so `gt/nps`
belongs to Dogfight and Purdue v1 lives elsewhere.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "tools" / "make_dataset_external.py"


def _gt_output_dirs() -> dict[str, list[str]]:
    """Map each build_*_gt* function to the `gt/<name>` directories it writes."""
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if not (node.name.startswith("build_") and "gt" in node.name):
            continue
        body = ast.get_source_segment(SRC.read_text(encoding="utf-8"), node) or ""
        out[node.name] = re.findall(r'OUT_ROOT\s*/\s*"gt"\s*/\s*"([^"]+)"', body)
    return out


def test_no_two_gt_builders_write_to_the_same_directory():
    dirs = _gt_output_dirs()
    assert dirs, "no GT builders found -- has the module been restructured?"

    owner: dict[str, str] = {}
    clashes = []
    for fn, targets in dirs.items():
        for t in targets:
            if t in owner:
                clashes.append(f"{t}: {owner[t]} and {fn}")
            owner[t] = fn
    assert not clashes, (
        "two GT builders share an output directory, and the directory IS the test set: "
        + "; ".join(clashes))


def test_the_nps_benchmark_gt_belongs_to_the_dogfight_builder():
    """`gt/nps` is what both arms are scored against, and the published NPS numbers we
    compare to (TransVisDrone 0.95, GLAD 0.89) are on Dogfight's re-annotations."""
    dirs = _gt_output_dirs()
    assert "nps" in dirs.get("build_nps_test_gt_dogfight", []), \
        "the Dogfight builder must own gt/nps"
    assert "nps" not in dirs.get("build_nps_test_gt", []), \
        "the Purdue v1 builder must NOT write to gt/nps -- it emits all 50 clips, 36 of " \
        "them training clips, under a different corner convention"


def test_the_purdue_builder_says_it_is_not_the_benchmark():
    """A future reader running --task nps-gt should not have to infer this from a path."""
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "build_nps_test_gt")
    doc = (ast.get_docstring(fn) or "").lower()
    assert "not the benchmark" in doc
    assert "dogfight" in doc
