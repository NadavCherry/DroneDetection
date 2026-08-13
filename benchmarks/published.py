"""Published results, with the protocol that produced each one.

The point of storing these is not convenience. It is that `tools/compare.py` can call
`Protocol.mismatches_with` and *derive* whether our number and theirs can be subtracted,
instead of relying on whoever writes the table to remember. Every previous
apples-to-oranges comparison in this repo happened because the protocol was in prose.

Provenance rules, enforced by `test_published.py`:

* `value` is transcribed from the source, never rounded, never converted between metrics.
* `verified=True` means the paper or repo page was opened during the 2026-08 sweep.
* A number reported by a *competing* paper about someone else's method is marked
  `reported_by_competitor=True`. Those get quoted constantly and are the least reliable
  class of number in this literature -- TransVisDrone's 0.15 on ARD100 comes from
  YOLOMG's authors, not from TransVisDrone's.
"""

from __future__ import annotations

from dataclasses import dataclass

from .protocol import (AP50, ARDMAV_GLAD, ARDMAV_GLAD_COMPLEX, ARDMAV_GLAD_ORDINARY,
                       ARDMAV_GLAD_SMALL, ARDMAV_MGMD, ARDMAV_OFFICIAL, DVB_OFFICIAL,
                       NPS_OFFICIAL, Protocol)


@dataclass(frozen=True)
class PublishedResult:
    method: str
    dataset_key: str
    metric: str                  # human-readable name of `value`
    value: float
    protocol: Protocol
    source_url: str
    year: int
    verified: bool
    reported_by_competitor: bool = False
    code_url: str = ""
    notes: str = ""


RESULTS: tuple[PublishedResult, ...] = (

    # ------------------------------------------------------------------ ARD-MAV
    # CORRECTED 2026-08-12 against the GLAD paper itself. This block previously held a
    # single row, "MGMD / GLAD, AP@0.25 = 0.55, on the official 15-video split", which
    # merged two papers and attached one's threshold to the other's split. The real bar is
    # 0.80 at IoU 0.5 -- higher, and at a threshold twice as strict.
    PublishedResult(
        method="GLAD", dataset_key="ardmav", metric="AP@0.5", value=0.80,
        protocol=ARDMAV_GLAD, year=2024, verified=True,
        source_url="https://arxiv.org/abs/2312.11008",
        code_url="https://github.com/WestlakeIntelligentRobotics/Global-Local-MAV-Detection",
        notes="THE BAR on the official 15-video split. P 0.92 / R 0.82 over 28,322 "
              "frames.\n"
              "WEIGHTS ARE OBTAINABLE -- mirrored 2026-08-12 to work/mirrors/glad, 113 MB: "
              "yolov5s_GLAD.pt (global, 14,359,541 B, sha256 9e8edde0...), "
              "yolov5s_GLAD-crop.pt (local, 14,454,645 B, 031813a6...), "
              "Net_best.pth (appearance classifier, 248,841 B, 0d3a4631...), plus prebuilt "
              "TensorRT .engine files. This is the only route to a PAIRED comparison "
              "against the incumbent, and therefore the only route to a p-value: a "
              "published scalar has no distribution, but a rival we run on our sequences "
              "does.\n"
              "TWO CAVEATS, both from the repo itself. (1) NO LICENCE FILE, so all rights "
              "are reserved by default: running it to produce comparison numbers is "
              "ordinary practice, vendoring it into this AGPL tree or shipping a "
              "derivative is not. (2) README, verbatim: 'This repository contains the "
              "basic codes for GLAD, the full codes with Kalman Filter, Adaptive Search "
              "Region, and other codes will be published in the future.' The release is "
              "not the published method, so a re-run measures GLAD-as-released and must "
              "be labelled that way -- never as a refutation of the paper's 0.80."),
    # The three condition rows carry their OWN split, so a per-condition number can never
    # be compared against an overall one. It happened: with the small row on
    # "official-test-15", the first run printed "ours higher" for our overall 0.754
    # against their small-subset 0.58, when our small-subset number is 0.530 and loses.
    PublishedResult(
        method="GLAD (small MAVs)", dataset_key="ardmav", metric="AP@0.5", value=0.58,
        protocol=ARDMAV_GLAD_SMALL, year=2024, verified=True,
        source_url="https://arxiv.org/abs/2312.11008",
        notes="P 0.82 / R 0.67 on phantom19,41,43,46,63. THE ROW TO CONTEST: GLAD's own "
              "weakest condition and the one a few-pixel temporal method is built for."),
    PublishedResult(
        method="GLAD (ordinary scenes)", dataset_key="ardmav", metric="AP@0.5", value=0.91,
        protocol=ARDMAV_GLAD_ORDINARY, year=2024, verified=True,
        source_url="https://arxiv.org/abs/2312.11008",
        notes="P 0.99 / R 0.96 on phantom09,10,30,47,70. Its easiest condition."),
    PublishedResult(
        method="GLAD (complex backgrounds)", dataset_key="ardmav", metric="AP@0.5",
        value=0.81, protocol=ARDMAV_GLAD_COMPLEX, year=2024, verified=True,
        source_url="https://arxiv.org/abs/2312.11008",
        notes="P 0.94 / R 0.86 on phantom05,08,58,65,86."),
    PublishedResult(
        method="TPH-YOLOv5l", dataset_key="ardmav", metric="AP@0.5", value=0.73,
        protocol=ARDMAV_GLAD, year=2024, verified=True,
        source_url="https://arxiv.org/abs/2312.11008",
        notes="Best non-GLAD entry in GLAD's own comparison table. Others there: "
              "YOLOv5s 0.61, MEGA 0.31, Dogfight 0.22."),
    PublishedResult(
        method="MGMD", dataset_key="ardmav", metric="AP@0.25", value=0.55,
        protocol=ARDMAV_MGMD, year=2024, verified=True,
        source_url="https://arxiv.org/abs/2410.10527",
        notes="NOT on the official split and NOT comparable to the GLAD rows above -- "
              "different threshold AND a split MGMD never enumerates. Kept so the number "
              "can be cited without implying it can be beaten on measurable ground."),

    # ------------------------------------------------------------------ ARD100
    PublishedResult(
        method="YOLOMG-1280", dataset_key="ard100", metric="AP@0.5", value=0.85,
        protocol=AP50, year=2025, verified=True,
        source_url="https://arxiv.org/abs/2503.07115",
        code_url="https://github.com/Irisky123/YOLOMG",
        notes="0.78 at 640 px input. Their own ablation: replacing the motion map with a "
              "second RGB stream collapses this to 0.33, and 3-frame differencing beats "
              "2-frame 0.78 vs 0.73 -- both of which support this project's thesis."),
    PublishedResult(
        method="YOLOv9", dataset_key="ard100", metric="AP@0.5", value=0.64,
        protocol=AP50, year=2025, verified=True, reported_by_competitor=True,
        source_url="https://arxiv.org/abs/2503.07115",
        notes="Best non-YOLOMG method in YOLOMG's own table."),
    PublishedResult(
        method="TransVisDrone", dataset_key="ard100", metric="AP@0.5", value=0.15,
        protocol=AP50, year=2023, verified=True, reported_by_competitor=True,
        source_url="https://arxiv.org/abs/2503.07115",
        notes="⚠ QUOTE WITH CARE. This is YOLOMG's re-run of a competitor, not "
              "TransVisDrone's own result. This repo previously reproduced it in bold as "
              "evidence of a rival 'collapsing'. Do not repeat that."),

    # ------------------------------------------------------------------ NPS-Drones
    PublishedResult(
        method="TransVisDrone", dataset_key="nps", metric="AP@0.5", value=0.95,
        protocol=NPS_OFFICIAL, year=2023, verified=True,
        source_url="https://arxiv.org/abs/2210.08423",
        code_url="https://github.com/tusharsangam/TransVisDrone",
        notes="Their own ablation is the useful part: a plain single-frame YOLOv5-l already "
              "scores 0.93, so the VideoSwin transformer is worth +0.02."),
    PublishedResult(
        method="YOLOMG-1280", dataset_key="nps", metric="AP@0.5", value=0.95,
        protocol=NPS_OFFICIAL, year=2025, verified=True,
        source_url="https://arxiv.org/abs/2503.07115", code_url="https://github.com/Irisky123/YOLOMG"),
    PublishedResult(
        method="Dogfight", dataset_key="nps", metric="AP@0.5", value=0.89,
        protocol=NPS_OFFICIAL, year=2021, verified=True,
        source_url="https://arxiv.org/abs/2103.17242",
        notes="~1 fps. TF 1.12 / CUDA 9 -- will not run on Blackwell, so it cannot be "
              "re-measured in-house; quote only."),

    # ------------------------------------------------------------------ Drone-vs-Bird
    PublishedResult(
        method="Laroca et al. (1st, 8th DvB Challenge)", dataset_key="dvb",
        metric="mean mAP@0.5", value=0.7390, protocol=DVB_OFFICIAL, year=2025, verified=True,
        source_url="https://arxiv.org/abs/2504.19347",
        notes="YOLOv11 + multi-scale tiling + drone/bird copy-paste + temporal consistency "
              "post-processing, over a 7-video self-chosen val split. Weakest videos: "
              "gopro_002 0.4491 and dji_phantom_4_hillside_cross 0.4992 -- the moving-camera "
              "cluttered cases, which are this project's design point. Their single-frame "
              "bird rejector FAILED ('lost many true positives') and their stated future work "
              "is a multi-frame patch classifier -- i.e. this repo's track classifier."),
    PublishedResult(
        method="YOLOMG (zero-shot)", dataset_key="dvb", metric="AP@0.5", value=0.41,
        protocol=DVB_OFFICIAL, year=2025, verified=True,
        source_url="https://arxiv.org/abs/2503.07115",
        notes="Precision 0.50, recall 0.47 — cross-dataset transfer with NO DvB training. "
              "Not comparable to Laroca's 0.7390, which trained on DvB. These two are "
              "frequently quoted side by side; they are different experiments."),

    # ------------------------------------------------------------------ DUT Anti-UAV
    PublishedResult(
        method="Lightweight YOLOv11 (P2, no P5)", dataset_key="dut_antiuav",
        metric="mAP@0.5", value=0.922, protocol=AP50, year=2026, verified=True,
        source_url="https://doi.org/10.3390/app16157423",
        notes="mAP@0.5:0.95 = 0.615, 2.11M params, Jetson Orin Nano Super + TensorRT FP16. "
              "Almost exactly this repo's edge recipe, independently derived. 'Add P2, delete "
              "P5' is a cheap ablation this repo has not tried. ⚠ MDPI landing page returned "
              "403 to automated fetch; numbers are from publisher-deposited metadata."),
    PublishedResult(
        method="UAV-DETR", dataset_key="dut_antiuav", metric="mAP@0.5:0.95", value=0.6715,
        protocol=AP50, year=2026, verified=True,
        source_url="https://arxiv.org/abs/2603.22841",
        code_url="https://github.com/wd-sir/UAVDETR",
        notes="mAP50 96.17. ⚠ Verified at commit e51d1d81: their NWD applies the pixel-unit "
              "constant C=12.8 to normalised [0,1] boxes, a scale error of exactly imgsz=640, "
              "which makes the NWD term effectively linear. Unreviewed preprint -- treat this "
              "as internal evidence that dronedet/nwd.py is right, not as a stick."),
)


def for_dataset(dataset_key: str) -> list[PublishedResult]:
    return sorted((r for r in RESULTS if r.dataset_key == dataset_key),
                  key=lambda r: -r.value)


def best_for_dataset(dataset_key: str, split: str | None = None) -> PublishedResult | None:
    """The number to beat ON ONE POPULATION. Excludes competitor-reported values.

    `split` matters, and defaults to the dataset's own official protocol rather than to
    "anything". A maximum taken across populations is not a bar: once GLAD's per-condition
    rows were added, the highest ARD-MAV value became 0.91 -- its score on the five EASIEST
    videos -- and this function offered it as the number to beat for a run scored over all
    fifteen. Different denominators, so the comparison flatters or damns at random
    depending on which subset happens to hold the maximum.

    `Protocol.mismatches_with` already refuses to subtract two such numbers. This makes the
    selection agree with that refusal instead of quietly handing over an incomparable row.
    """
    from .catalog import DATASETS
    if split is None:
        proto = getattr(DATASETS.get(dataset_key), "official_protocol", None)
        split = proto.split if proto is not None else None
    rs = [r for r in for_dataset(dataset_key) if not r.reported_by_competitor]
    if split is not None:
        on_split = [r for r in rs if r.protocol.split == split]
        # Fall back rather than return nothing. Several datasets record no split on their
        # published rows at all (ard100's are ''), and for those the filter cannot tell a
        # subset from the whole -- so it has nothing to protect against, and erasing the
        # bar would be a worse answer than the unfiltered one. Where splits ARE recorded,
        # the filter does its job.
        rs = on_split or rs
    return rs[0] if rs else None
