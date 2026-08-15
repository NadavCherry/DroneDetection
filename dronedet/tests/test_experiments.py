"""Tests for the experiment registry and `tools/train.py`.

These exist because of the failure they are designed to make impossible: a checkpoint in
`work/runs/` that nobody can trace to the code, the data or the seed that produced it.
The three properties worth guarding hardest are therefore

1. **the manifest is written before the first batch** -- a run that dies in epoch 3 must
   still be traceable, so `test_manifest_is_written_before_the_trainer_is_called` calls a
   fake trainer that asserts the file already exists;
2. **the seed reaches the trainer** and appears in both the run directory name and the
   manifest, because "three seeds" is a claim about three distinct runs;
3. **the labels on disk match the config's claim about them** -- the ARD-MAV builder
   writes the inflated and the true-extent datasets to the same directory, so the
   headline experiment can silently be run against the control's data.

Nothing here imports torch or ultralytics: `tools/train.py` keeps every heavy import
inside `_ultralytics_trainer`, and these tests substitute a fake for it. That is the
constraint CI enforces, and importing this module at all is part of the test.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from benchmarks.protocol import ARDMAV_OFFICIAL  # noqa: E402
from configs.experiments import (Augmentation, EXPERIMENTS, GROUPS,  # noqa: E402
                                 ExperimentConfig, NO_PHOTOMETRIC_AUG, UAV_DETR_B4_640,
                                 YOLOV8S_P2_MEASURED, check_strides, describe_all, get,
                                 names, resolve)


def _load_train_module():
    """Load tools/train.py under a unique name (tools/ is not a package)."""
    path = REPO / "tools" / "train.py"
    spec = importlib.util.spec_from_file_location("specklock_tools_train", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


T = _load_train_module()


# --------------------------------------------------------------------------- fixtures
def _write_dataset(root: Path, min_side: float, tile: int = 640,
                   n_train: int = 24, n_val: int = 8) -> Path:
    """A minimal YOLO dataset whose labels really carry the given inflation."""
    true_sides = [3.0, 4.5, 6.0, 9.0, 14.0, 20.0]
    for split, n in (("train", n_train), ("val", n_val)):
        img = root / "images" / split
        lbl = root / "labels" / split
        img.mkdir(parents=True, exist_ok=True)
        lbl.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            s = max(true_sides[i % len(true_sides)], min_side)
            (img / f"f{i:04d}.jpg").write_bytes(b"\xff\xd8fake-jpeg" + bytes([i % 251]))
            (lbl / f"f{i:04d}.txt").write_text(
                f"0 0.500000 0.500000 {s / tile:.6f} {s / tile:.6f}\n", encoding="utf-8")
    (root / "data.yaml").write_text(
        f"path: {root.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: drone\n", encoding="utf-8")
    # Real builders record what they built (tools/make_dataset_external.write_data_yaml),
    # and the inflation check trusts that record over any inference from box sizes. A
    # fixture without it would exercise only the legacy path.
    (root / "BUILD.json").write_text(
        json.dumps({"task": "test-fixture", "min_side": float(min_side), "tile": tile}),
        encoding="utf-8")
    return root / "data.yaml"


def _args(**over):
    """An argparse.Namespace straight from the real parser, so defaults stay in one place."""
    argv = ["--config", "baseline_ardmav"]
    for k, v in over.items():
        flag = "--" + k.replace("_", "-")
        if v is True:
            argv.append(flag)
        elif v is not False and v is not None:
            argv += [flag, str(v)]
    return T.build_parser().parse_args(argv)


@pytest.fixture
def demo_cfg(tmp_path):
    """A runnable config pointing at a real (tiny) dataset, inflation 12 px."""
    data_yaml = _write_dataset(tmp_path / "ds", min_side=12.0)
    return ExperimentConfig(
        name="demo", datasets=("ardmav",), data=str(data_yaml),
        model_cfg="yolov8s-p2.yaml", weights="yolov8s.pt",
        imgsz=640, batch=2, epochs=3, seed=7, min_side=12.0, tile_px=640,
        aug=Augmentation(hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, close_mosaic=0),
        protocol_key="ardmav-official", build_command="echo build me",
        notes="a fixture, not an experiment")


# ================================================================= registry & validation
def test_every_registered_experiment_is_named_after_itself():
    for key, cfg in EXPERIMENTS.items():
        assert key == cfg.name


def test_the_five_promised_experiments_are_registered():
    for n in ("baseline_ardmav", "trueextent_ardmav", "p2_no_p5_ardmav", "birds_2class",
              "temporal_stack_ablation_single", "temporal_stack_ablation_stack"):
        assert n in EXPERIMENTS, n
    assert "temporal_stack_ablation" in GROUPS


def test_every_config_states_a_rationale_and_how_its_data_was_built():
    for cfg in EXPERIMENTS.values():
        assert cfg.notes.strip(), f"{cfg.name} has no rationale"
        assert cfg.build_command or cfg.missing, f"{cfg.name} cannot say how data was built"


def test_describe_all_lists_every_experiment_and_group():
    text = describe_all()
    for n in names():
        assert n in text
    for g in GROUPS:
        assert g in text


def test_unknown_experiment_name_names_the_alternatives():
    with pytest.raises(KeyError) as e:
        get("no_such_experiment")
    assert "baseline_ardmav" in str(e.value)


@pytest.mark.parametrize("bad, needle", [
    (dict(imgsz=650), "multiple of 32"),
    (dict(batch=0), "batch"),
    (dict(epochs=0), "epochs"),
    (dict(datasets=()), "datasets is empty"),
    (dict(datasets=("not_a_dataset",)), "unknown dataset key"),
    (dict(notes=""), "rationale"),
    (dict(build_command=""), "build_command"),
    (dict(nwd_assign_ratio=1.5), "nwd_assign_ratio"),
    (dict(nwd_loss_c=0.0), "nwd_loss_c"),
    (dict(min_side=640.0), "min_side"),
    (dict(min_side=-1.0), "min_side"),
    (dict(stack_dt=6), "stack_dt is set but temporal_stack is False"),
])
def test_invalid_configs_are_rejected_at_construction(bad, needle):
    base = dict(name="t", datasets=("ardmav",), data="d.yaml", model_cfg="m.yaml",
                weights="w.pt", notes="why", build_command="build")
    base.update(bad)
    with pytest.raises(ValueError) as e:
        ExperimentConfig(**base)
    assert needle in str(e.value)


def test_a_temporal_stack_may_not_carry_hsv_jitter():
    """Hue/saturation jitter on a stack of moments remixes *time*, not colour. The v3
    rounds passed 0 0 by hand; this makes it impossible to forget."""
    with pytest.raises(ValueError) as e:
        ExperimentConfig(name="t", datasets=("ardmav",), data="d.yaml",
                         model_cfg="m.yaml", weights="w.pt", notes="why",
                         build_command="b", temporal_stack=True, stack_dt=3)
    assert "remixes the moments" in str(e.value)


def test_a_temporal_stack_must_state_its_channel_spacing():
    with pytest.raises(ValueError) as e:
        ExperimentConfig(name="t", datasets=("ardmav",), data="d.yaml",
                         model_cfg="m.yaml", weights="w.pt", notes="why",
                         build_command="b", temporal_stack=True,
                         aug=NO_PHOTOMETRIC_AUG)
    assert "stack_dt" in str(e.value)


def test_mosaic_that_is_closed_before_it_opens_is_rejected():
    with pytest.raises(ValueError) as e:
        ExperimentConfig(name="t", datasets=("ardmav",), data="d.yaml",
                         model_cfg="m.yaml", weights="w.pt", notes="why",
                         build_command="b", epochs=5)
    assert "close_mosaic" in str(e.value)


def test_a_non_benchmark_corpus_must_declare_itself_local():
    cfg = get("temporal_stack_ablation_single")
    assert "local:07_05" in cfg.datasets
    assert cfg.external_dataset_keys == ("ardmav", "nps"), \
        "only catalogued datasets may back an external claim"


def test_overrides_are_revalidated():
    cfg = get("baseline_ardmav")
    assert cfg.with_overrides(batch=4).batch == 4
    with pytest.raises(ValueError):
        cfg.with_overrides(imgsz=641)


# ============================================================ what the experiments claim
def test_the_baseline_carries_an_iou_ceiling_that_the_true_extent_run_does_not():
    """The headline experiment, expressed as a protocol difference rather than as prose:
    a 12 px inflated label caps IoU independently of the detector, and `mismatches_with`
    is what says so at comparison time."""
    base = get("baseline_ardmav").protocol()
    true = get("trueextent_ardmav").protocol()
    assert base.label_inflation_px == 12.0
    assert true.label_inflation_px is None
    assert any("inflated" in m for m in base.mismatches_with(ARDMAV_OFFICIAL))
    assert true.mismatches_with(ARDMAV_OFFICIAL) == []


def test_true_extent_differs_from_the_baseline_only_where_the_hypothesis_says_it_should():
    a, b = get("baseline_ardmav"), get("trueextent_ardmav")
    differing = {f.name for f in a.__dataclass_fields__.values()
                 if getattr(a, f.name) != getattr(b, f.name)}
    assert differing <= {"name", "min_side", "nwd", "notes", "tags", "build_command"}, \
        f"unexpected second variable in the headline experiment: {differing}"


def test_the_no_p5_experiment_expects_a_three_level_head():
    """The yaml was instantiated once (strides (4, 8, 16), 7.41M params vs 10.88M) and is
    no longer flagged unverified. CI has no torch, so what is guarded here is the file
    itself: a later edit that re-adds the P5 branch has to change this line too."""
    cfg = get("p2_no_p5_ardmav")
    assert cfg.expected_strides == (4, 8, 16)
    assert not cfg.unverified_model_cfg

    text = (REPO / cfg.model_cfg).read_text(encoding="utf-8")
    detect = [ln for ln in text.splitlines() if "Detect" in ln and not ln.lstrip().startswith("#")]
    assert detect == ["  - [[18, 21, 24], 1, Detect, [nc]] # Detect(P2, P3, P4) -- no P5"], detect
    assert len(cfg.expected_strides) == 3, "three strides, three Detect inputs"


def test_the_ablation_arms_differ_only_in_their_input_representation():
    single, stack = resolve("temporal_stack_ablation")
    differing = {f.name for f in single.__dataclass_fields__.values()
                 if getattr(single, f.name) != getattr(stack, f.name)}
    assert differing <= {"name", "data", "temporal_stack", "stack_dt", "notes",
                         "build_command"}, differing
    assert single.aug == stack.aug, "augmentation must not be a second variable"
    assert single.imgsz == stack.imgsz and single.epochs == stack.epochs


def test_a_blocked_experiment_refuses_to_run_and_says_what_is_missing():
    cfg = get("birds_2class")
    assert not cfg.runnable
    with pytest.raises(NotImplementedError) as e:
        cfg.require_runnable()
    assert "make_dataset_birds" in str(e.value)


def test_group_resolution_expands_and_single_names_do_not():
    assert [c.name for c in resolve("temporal_stack_ablation")] == \
        list(GROUPS["temporal_stack_ablation"])
    assert [c.name for c in resolve("baseline_ardmav")] == ["baseline_ardmav"]


# ================================================================== the 8 GB VRAM guard
def test_the_shipped_configs_fit_an_8gb_card():
    for cfg in EXPERIMENTS.values():
        assert cfg.vram_warning(8.0) is None, f"{cfg.name}: {cfg.vram_warning(8.0)}"


def test_an_oversized_batch_warns_and_quotes_its_measured_references():
    """The warning must say what it reasoned from and whether anyone has observed it.

    This assertion used to be `"UNMEASURED" in w` -- correct while the YOLO anchor was a
    guess extrapolated from a DETR's batch-4 figure. yolov8s-p2 @ 640 px was measured
    directly on 2026-08-12 (3.73 / 7.08 / 10.39 GiB at batch 8 / 16 / 32), so it flips: a
    yolov8s-p2 config must now be warned about on yolov8s-p2 evidence and must say so.
    Both anchors stay in the message, because a reader checking the arithmetic needs them.
    """
    cfg = get("baseline_ardmav").with_overrides(batch=64, imgsz=1280)
    w = cfg.vram_warning(8.0)
    assert w is not None
    assert "LIKELY OOM" in w
    assert f"{UAV_DETR_B4_640.gib:.2f}" in w and "batch 4" in w
    assert f"{YOLOV8S_P2_MEASURED.gib:.2f}" in w
    assert "[MEASURED]" in w and "UNMEASURED" not in w


def test_the_suggested_batch_actually_fits():
    cfg = get("baseline_ardmav").with_overrides(batch=64, imgsz=1280)
    b = cfg.max_batch_for(8.0)
    assert b >= 1
    assert cfg.with_overrides(batch=b).vram_warning(8.0) is None


def test_vram_estimate_is_linear_in_batch_and_quadratic_in_side():
    cfg = get("baseline_ardmav")
    assert cfg.estimated_vram_gib(batch=8) == pytest.approx(
        2 * cfg.estimated_vram_gib(batch=4))
    assert cfg.estimated_vram_gib(imgsz=1280) == pytest.approx(
        4 * cfg.estimated_vram_gib(imgsz=640))


# ======================================================================== stride check
def test_check_strides_passes_the_expected_head_and_names_the_mismatch():
    assert check_strides([4, 8, 16, 32], (4, 8, 16, 32)) is None
    msg = check_strides([8, 16, 32], (4, 8, 16))
    assert msg and "[8, 16, 32]" in msg and "[4, 8, 16]" in msg


# ==================================================================== data.yaml parsing
def test_parse_data_yaml_reads_what_the_repo_writes(tmp_path):
    y = _write_dataset(tmp_path / "ds", min_side=12.0)
    d = T.parse_data_yaml(y)
    assert d["train"] == "images/train" and d["val"] == "images/val"
    assert d["names"] == {0: "drone"}


def test_parse_data_yaml_refuses_a_shape_it_does_not_understand(tmp_path):
    y = tmp_path / "bad.yaml"
    y.write_text("train: images/train\nval: images/val\n  stray: value\n", encoding="utf-8")
    with pytest.raises(ValueError):
        T.parse_data_yaml(y)


def test_missing_split_key_is_an_error(tmp_path):
    y = tmp_path / "bad.yaml"
    y.write_text("train: images/train\n", encoding="utf-8")
    with pytest.raises(ValueError):
        T.parse_data_yaml(y)


# =========================================================== the label-inflation check
def test_label_stats_recover_the_inflation_that_built_the_dataset(tmp_path):
    _write_dataset(tmp_path / "inf", min_side=12.0)
    _write_dataset(tmp_path / "true", min_side=0.0)
    inflated = T.label_box_stats(tmp_path / "inf" / "labels" / "train", 640)
    true = T.label_box_stats(tmp_path / "true" / "labels" / "train", 640)
    assert inflated["min_side_px"] == pytest.approx(12.0, abs=0.01)
    assert true["min_side_px"] == pytest.approx(3.0, abs=0.01)


def test_true_extent_config_against_an_inflated_dataset_is_refused(tmp_path):
    """The exact trap: both ARD-MAV builds land in the same directory, so the headline
    experiment can be run against the control's labels and produce a plausible curve."""
    stats = T.label_box_stats(_write_dataset(tmp_path / "inf", 12.0).parent
                              / "labels" / "train", 640)
    msg = T.check_label_inflation(stats, min_side=0.0)
    assert msg and "TRUE EXTENTS" in msg


def test_inflated_config_against_a_true_extent_dataset_is_refused(tmp_path):
    stats = T.label_box_stats(_write_dataset(tmp_path / "true", 0.0).parent
                              / "labels" / "train", 640)
    msg = T.check_label_inflation(stats, min_side=12.0)
    assert msg and "smaller --min-side" in msg


def test_a_large_true_extent_corpus_is_not_mistaken_for_an_inflated_one(tmp_path):
    """NPS-Drones, the regression. Its true-extent boxes are 10-25 px with integer
    corners, so its smallest box is 10.00 -- larger than ARD-MAV's and not inflated.

    The old rule rejected exactly this: it refused any true-extent build whose smallest
    sampled box was >= 8 px, a threshold calibrated on ARD-MAV alone. Six NPS training
    jobs died in two seconds against a perfectly good dataset. A guard that fires on a
    correct dataset teaches people to override guards, which is worse than no guard.
    """
    root = tmp_path / "nps_like"
    for split, n in (("train", 24), ("val", 8)):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            side = [10.0, 11.0, 12.0, 14.0, 18.0, 25.0][i % 6]   # integer-quantised, all >= 10
            (root / "images" / split / f"f{i:04d}.jpg").write_bytes(b"\xff\xd8x" + bytes([i % 251]))
            (root / "labels" / split / f"f{i:04d}.txt").write_text(
                f"0 0.5 0.5 {side/640:.6f} {side/640:.6f}\n", encoding="utf-8")
    (root / "BUILD.json").write_text(json.dumps({"min_side": 0.0}), encoding="utf-8")

    stats = T.label_box_stats(root / "labels" / "train", 640)
    assert stats["min_side_px"] >= 10.0, "fixture should model NPS's larger targets"
    assert T.check_label_inflation(stats, min_side=0.0) is None


def test_a_dataset_with_no_build_record_warns_rather_than_refusing(tmp_path, capsys):
    """Legacy datasets predate BUILD.json, and inflation is NOT decidable from the box
    distribution: `--min-side 12` is a floor whose spike is partial, and a genuinely
    small-target corpus can pile up at its minimum through integer quantisation. The two
    overlap. So without the record, say so and continue rather than guess and block."""
    root = _write_dataset(tmp_path / "legacy", 12.0).parent
    (root / "BUILD.json").unlink()
    stats = T.label_box_stats(root / "labels" / "train", 640)
    assert stats.get("build_min_side") is None
    assert T.check_label_inflation(stats, min_side=0.0) is None      # no rejection
    assert "no BUILD.json" in capsys.readouterr().err                # but it is said


def test_matching_inflation_passes_both_ways(tmp_path):
    inf = T.label_box_stats(_write_dataset(tmp_path / "i", 12.0).parent / "labels" / "train", 640)
    tru = T.label_box_stats(_write_dataset(tmp_path / "t", 0.0).parent / "labels" / "train", 640)
    assert T.check_label_inflation(inf, 12.0) is None
    assert T.check_label_inflation(tru, 0.0) is None


def test_an_absent_label_dir_does_not_invent_a_verdict():
    stats = T.label_box_stats(Path("/nonexistent/labels"), 640)
    assert stats["present"] is False
    assert T.check_label_inflation(stats, 12.0) is None


def test_a_dataset_with_the_wrong_number_of_classes_is_refused(tmp_path):
    """Ultralytics reads `nc` from the data.yaml and never consults the config, so a
    two-class experiment pointed at a one-class build trains a healthy single-class
    detector and files it under a name that promises a bird false-alarm rate."""
    _write_dataset(tmp_path / "ds", min_side=12.0)
    parsed = {"names": {0: "drone"}}
    stats = T.label_box_stats(tmp_path / "ds" / "labels" / "train", 640)
    assert T.check_class_count(parsed, stats, ("drone",)) is None
    msg = T.check_class_count(parsed, stats, ("drone", "bird"))
    assert msg and "declares 1 class" in msg


def test_a_label_naming_a_class_the_config_does_not_have_is_refused(tmp_path):
    root = tmp_path / "ds"
    _write_dataset(root, min_side=12.0)
    (root / "labels" / "train" / "f0000.txt").write_text("3 0.5 0.5 0.02 0.02\n", encoding="utf-8")
    stats = T.label_box_stats(root / "labels" / "train", 640)
    msg = T.check_class_count({"names": {0: "drone"}}, stats, ("drone",))
    assert msg and "[3]" in msg


def test_data_override_is_refused_for_a_group_whose_arms_use_different_datasets(tmp_path):
    """`--data` applies to every arm; on the temporal ablation that does not override the
    experiment, it deletes it -- the arms differ in nothing else."""
    with pytest.raises(SystemExit):
        T.main(["--config", "temporal_stack_ablation", "--data", str(tmp_path / "x.yaml"),
                "--dry-run"])


def test_dataset_manifest_hashes_both_splits_and_changes_with_the_data(tmp_path):
    y = _write_dataset(tmp_path / "ds", min_side=12.0)
    m1 = T.dataset_manifest(y, tile_px=640)
    assert set(m1["splits"]) == {"train", "val"}
    assert m1["splits"]["train"]["images"]["n_files"] == 24
    assert m1["splits"]["train"]["labels"]["listing_sha256"]

    (tmp_path / "ds" / "labels" / "train" / "f0000.txt").write_text(
        "0 0.5 0.5 0.05 0.05\n0 0.2 0.2 0.05 0.05\n", encoding="utf-8")
    m2 = T.dataset_manifest(y, tile_px=640)
    assert (m1["splits"]["train"]["labels"]["listing_sha256"]
            != m2["splits"]["train"]["labels"]["listing_sha256"])


# ============================================================== the trainer kwarg seam
def test_the_seed_reaches_the_trainer_and_the_run_directory(tmp_path):
    cfg = get("baseline_ardmav")
    kw = T.ultralytics_kwargs(cfg, 3, tmp_path / "runs" / cfg.run_name(3))
    assert kw["seed"] == 3
    assert kw["deterministic"] is True
    assert kw["name"] == "baseline_ardmav-s3"
    assert Path(kw["project"]).is_absolute()


def test_exist_ok_is_set_so_the_manifest_and_the_weights_share_a_directory(tmp_path):
    kw = T.ultralytics_kwargs(get("baseline_ardmav"), 0, tmp_path / "runs" / "x")
    assert kw["exist_ok"] is True


def test_optional_knobs_are_omitted_rather_than_passed_as_none(tmp_path):
    kw = T.ultralytics_kwargs(get("baseline_ardmav"), 0, tmp_path / "r")
    assert "lr0" not in kw and "freeze" not in kw and "device" not in kw
    kw2 = T.ultralytics_kwargs(get("baseline_ardmav").with_overrides(lr0=0.002), 0,
                               tmp_path / "r", device="0")
    assert kw2["lr0"] == 0.002 and kw2["device"] == "0"


def test_the_temporal_arm_reaches_the_trainer_with_photometric_jitter_off(tmp_path):
    kw = T.ultralytics_kwargs(get("temporal_stack_ablation_stack"), 0, tmp_path / "r")
    assert kw["hsv_h"] == 0 and kw["hsv_s"] == 0 and kw["hsv_v"] == 0


def test_seeds_for_runs_consecutive_seeds_from_the_configs_own_base():
    cfg = get("baseline_ardmav")
    assert T.seeds_for(cfg, _args(seeds=3)) == [cfg.seed, cfg.seed + 1, cfg.seed + 2]
    assert T.seeds_for(cfg, _args(seed=100, seeds=3)) == [100, 101, 102]
    assert T.seeds_for(cfg, _args()) == [cfg.seed]


def test_set_seeds_records_what_it_actually_did():
    rec = T.set_seeds(11)
    assert rec["seed"] == 11
    assert rec["python_random"] is True
    assert rec["env"]["PYTHONHASHSEED"] == "11"
    assert "numpy" in rec and "torch" in rec       # value may be a 'not seeded' reason


def test_seeding_makes_the_python_rng_reproducible():
    import random
    T.set_seeds(5)
    a = [random.random() for _ in range(4)]
    T.set_seeds(5)
    assert [random.random() for _ in range(4)] == a


# ========================================================================== the manifest
def test_manifest_records_the_provenance_a_checkpoint_needs(tmp_path, demo_cfg):
    run_dir = tmp_path / "runs" / demo_cfg.run_name(7)
    ds = T.dataset_manifest(Path(demo_cfg.data), demo_cfg.tile_px)
    kw = T.ultralytics_kwargs(demo_cfg, 7, run_dir)
    m = T.build_manifest(demo_cfg, 7, run_dir, ds, kw, T.set_seeds(7), None,
                         argv=["tools/train.py", "--config", "demo", "--seeds", "3"])

    assert m["seed"] == 7 and m["run_name"] == "demo-s7"
    assert m["config"]["min_side"] == 12.0
    assert m["config"]["aug"]["hsv_h"] == 0.0            # nested dataclass survives
    assert m["command"]["shell"] == "tools/train.py --config demo --seeds 3"
    assert m["command"]["cwd"] and m["command"]["executable"]
    assert set(m["git"]) >= {"sha", "dirty", "branch", "uncommitted_diff_sha256"}
    assert set(m["packages"]) >= {"torch", "ultralytics", "numpy"}
    assert "available" in m["gpu"]
    assert m["dataset"]["splits"]["train"]["labels"]["listing_sha256"]
    assert m["protocol"]["fields"]["label_inflation_px"] == 12.0
    assert m["vram"]["estimate_gib"] > 0
    assert m["trainer_kwargs"]["seed"] == 7
    assert m["status"] == "started"


def test_manifest_is_json_serialisable_with_no_custom_encoder(tmp_path, demo_cfg):
    m = T.build_manifest(demo_cfg, 0, tmp_path / "r", {"present": False},
                         T.ultralytics_kwargs(demo_cfg, 0, tmp_path / "r"),
                         T.set_seeds(0), None, argv=["x"])
    json.loads(json.dumps(m, default=str))


def test_git_provenance_reports_this_repo():
    g = T.git_provenance(REPO)
    if g["sha"] is None:
        pytest.skip("no git available")
    assert len(g["sha"]) == 40
    assert isinstance(g["dirty"], bool)


def test_git_output_that_cannot_be_read_is_unknown_and_not_empty(monkeypatch):
    """`out.stdout` can be None while returncode is 0, and it must not become "".

    Windows `subprocess.run(capture_output=True, timeout=...)` reads the pipes on helper
    threads; under heavy disk load one can die and hand back None. Observed for real while
    a 40,000-tile build saturated the disk -- `out.stdout.strip()` raised AttributeError
    inside build_manifest() and took eight tests with it.

    None must mean "unknown", never "empty": `git status --porcelain` returning "" means
    the tree is CLEAN, so collapsing an unreadable result to "" would stamp a manifest with
    clean provenance for a dirty tree -- silently, and exactly on the checkpoints produced
    under load.
    """
    class _Broken:
        returncode = 0
        stdout = None
        stderr = ""

    monkeypatch.setattr(T.subprocess, "run", lambda *a, **k: _Broken())
    assert T._run_git(REPO, "status", "--porcelain") is None


def test_provenance_survives_git_being_unreadable(monkeypatch):
    """A training run must not die because provenance bookkeeping could not read git."""
    monkeypatch.setattr(T, "_run_git", lambda *a, **k: None)
    g = T.git_provenance(REPO)
    assert g["sha"] is None
    assert g["dirty"] is False and g["dirty_file_count"] == 0
    assert g["uncommitted_diff_sha256"] is None
    json.dumps(g)                                   # still serialisable into the manifest


# ============================================================= train_one, with a fake GPU
def test_manifest_is_written_before_the_trainer_is_called(tmp_path, demo_cfg):
    """A run that crashes in epoch 3 must still be traceable. This is the whole point."""
    seen = {}

    def fake(cfg, seed, run_dir, kwargs):
        seen["manifest_existed"] = (run_dir / "MANIFEST.json").exists()
        seen["seed"] = seed
        seen["kwargs"] = kwargs
        return {"fake": True}

    run_dir = T.train_one(demo_cfg, 7, _args(out_root=tmp_path / "runs"), trainer=fake)
    assert seen["manifest_existed"] is True
    assert seen["seed"] == 7
    assert seen["kwargs"]["seed"] == 7
    assert run_dir.name == "demo-s7"

    m = json.loads((run_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    assert m["experiment_name"] == "demo"
    r = json.loads((run_dir / "RESULT.json").read_text(encoding="utf-8"))
    assert r["status"] == "ok" and r["trainer"] == {"fake": True}


def test_a_crashed_run_still_leaves_a_manifest_and_a_failure_record(tmp_path, demo_cfg):
    def boom(cfg, seed, run_dir, kwargs):
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError):
        T.train_one(demo_cfg, 1, _args(out_root=tmp_path / "runs"), trainer=boom)
    run_dir = tmp_path / "runs" / "demo-s1"
    assert (run_dir / "MANIFEST.json").exists()
    r = json.loads((run_dir / "RESULT.json").read_text(encoding="utf-8"))
    assert r["status"] == "failed" and "out of memory" in r["error"]


def test_dry_run_records_everything_and_trains_nothing(tmp_path, demo_cfg):
    def fake(*a, **k):
        raise AssertionError("the trainer must not be called on a dry run")

    run_dir = T.train_one(demo_cfg, 0, _args(out_root=tmp_path / "runs", dry_run=True),
                          trainer=fake)
    assert (run_dir / "MANIFEST.json").exists()
    assert not (run_dir / "RESULT.json").exists()


def test_an_already_recorded_run_is_not_silently_overwritten(tmp_path, demo_cfg):
    args = _args(out_root=tmp_path / "runs")
    T.train_one(demo_cfg, 0, args, trainer=lambda *a: {})
    with pytest.raises(SystemExit) as e:
        T.train_one(demo_cfg, 0, args, trainer=lambda *a: {})
    assert "already" in str(e.value)
    T.train_one(demo_cfg, 0, _args(out_root=tmp_path / "runs", force=True),
                trainer=lambda *a: {})


def test_training_refuses_to_start_against_the_wrong_dataset_build(tmp_path):
    """A true-extent config pointed at an inflated dataset aborts, and the abort names
    the build command that would fix it."""
    data_yaml = _write_dataset(tmp_path / "ds", min_side=12.0)
    cfg = get("trueextent_ardmav").with_overrides(data=str(data_yaml))
    with pytest.raises(SystemExit) as e:
        T.train_one(cfg, 0, _args(out_root=tmp_path / "runs"), trainer=lambda *a: {})
    assert "TRUE EXTENTS" in str(e.value)
    assert "--min-side 0" in str(e.value)


def test_a_missing_dataset_aborts_with_the_command_that_builds_it(tmp_path):
    cfg = get("baseline_ardmav").with_overrides(data=str(tmp_path / "nope" / "data.yaml"))
    with pytest.raises(SystemExit) as e:
        T.train_one(cfg, 0, _args(out_root=tmp_path / "runs"), trainer=lambda *a: {})
    assert "make_dataset_external.py" in str(e.value)


def test_a_blocked_experiment_cannot_be_launched(tmp_path):
    with pytest.raises(NotImplementedError):
        T.train_one(get("birds_2class"), 0, _args(out_root=tmp_path / "runs"),
                    trainer=lambda *a: {})


def test_seeds_three_writes_three_independent_run_directories(tmp_path, demo_cfg):
    args = _args(out_root=tmp_path / "runs", seeds=3)
    for seed in T.seeds_for(demo_cfg, args):
        T.train_one(demo_cfg, seed, args, trainer=lambda *a: {"ok": True})
    dirs = sorted(p.name for p in (tmp_path / "runs").iterdir())
    assert dirs == ["demo-s7", "demo-s8", "demo-s9"]
    seeds = [json.loads((tmp_path / "runs" / d / "MANIFEST.json").read_text(encoding="utf-8"))["seed"]
             for d in dirs]
    assert seeds == [7, 8, 9]


def test_main_lists_the_registry_without_touching_a_gpu(capsys):
    assert T.main(["--list"]) == 0
    assert "baseline_ardmav" in capsys.readouterr().out
