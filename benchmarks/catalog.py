"""The dataset registry: what exists, how to get it, and what it can prove.

A registry rather than prose, because three things about a dataset decide whether a
number computed on it means anything, and all three were previously only in someone's
head: its **official split**, its **official protocol**, and which **conditions** it
actually contains. The last one is why this project could not previously say anything
about rain or night -- not because the models failed, but because no dataset on disk had
a rainy frame in it.

`Condition` is a first-class field for exactly that reason. `tools/evaluate.py` groups
its scorecard by condition, so "works at night" becomes a row rather than a hope.

Stdlib only (dataclasses), so it imports in the torch-free CI job and needs no YAML.
Every entry carries `verified`: True means the download route was opened and confirmed
during the 2026-08 research sweep; False means it came from a paper or a registry record
and must be checked before it is relied on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .protocol import ARDMAV_OFFICIAL, DVB_OFFICIAL, NPS_OFFICIAL, Protocol


class Gate(str, Enum):
    """What stands between you and the bytes. Drives `tools/fetch_data.py`."""
    OPEN = "open"                 # direct download, no interaction
    GDRIVE = "gdrive"             # Google Drive, needs gdown for the confirm token
    FORM = "form"                 # a web form, one-off manual step
    AGREEMENT = "agreement"       # signed data-usage agreement, days of latency
    BAIDU = "baidu"               # BaiduYun, effectively manual outside China
    UNKNOWN = "unknown"


class Condition(str, Enum):
    """Operating conditions a dataset actually contains, for stratified reporting."""
    CLEAR = "clear"
    NIGHT = "night"
    FOG = "fog"                   # includes haze
    RAIN = "rain"
    SNOW = "snow"
    CLOUD_BACKGROUND = "cloud"    # target against/into cloud
    LOW_CONTRAST = "low-contrast"
    MOVING_CAMERA = "moving-camera"
    CLUTTER = "clutter"           # trees, buildings, ground texture


@dataclass(frozen=True)
class Dataset:
    key: str
    name: str
    year: int
    modality: str                        # 'rgb-video' | 'rgb-stills' | 'ir-video' | 'rgbt' | 'event'
    url: str
    licence: str
    gate: Gate
    classes: tuple[str, ...]
    conditions: tuple[Condition, ...]
    verified: bool
    frames: int | None = None
    sequences: int | None = None
    resolution: str = ""
    target_px_median: float | None = None
    size_gb: float | None = None
    download_id: str = ""                # gdrive file id, zenodo doi, or direct URL
    official_protocol: Protocol | None = None
    official_test: tuple[str, ...] = ()  # sequence ids, when the split is published
    official_val: tuple[str, ...] = ()
    has_birds: bool = False
    priority: int = 99                   # acquisition order from the research briefing
    notes: str = ""

    @property
    def is_video(self) -> bool:
        """Only video can carry a temporal method. Stills are for the single-frame
        branch and the loss/matching ablations -- never as evidence for the temporal claim."""
        return "video" in self.modality

    @property
    def usable_offline(self) -> bool:
        return self.gate in (Gate.OPEN, Gate.GDRIVE)


ARD_TEST = tuple(f"phantom{n:02d}" for n in
                 (5, 8, 9, 10, 19, 30, 41, 43, 46, 47, 58, 63, 65, 70, 86))
ARD_VAL = ("phantom06", "phantom23", "phantom45", "phantom61", "phantom79")


DATASETS: dict[str, Dataset] = {d.key: d for d in [

    # ---------------------------------------------------------------- priority 1: on disk
    Dataset(
        key="ardmav", name="ARD-MAV", year=2024, modality="rgb-video",
        url="https://github.com/WestlakeIntelligentRobotics/Global-Local-MAV-Detection",
        licence="MIT", gate=Gate.GDRIVE, verified=True,
        download_id="1_I5jR-a-Jlan96s7XD3QeLLddb51rDT_",
        classes=("mav",), conditions=(Condition.MOVING_CAMERA, Condition.CLUTTER, Condition.CLEAR),
        frames=107497, sequences=60, resolution="1920x1080", size_gb=14.6,
        target_px_median=11.8, official_protocol=ARDMAV_OFFICIAL,
        official_test=ARD_TEST, official_val=ARD_VAL, priority=1,
        notes="ACQUIRED and measured 2026-08-12: 106,456 boxes, median sqrt(area) 11.8 px, "
              "21.3% very-tiny / 43.8% tiny. Published SOTA is only AP 0.55 (MGMD, IoU 0.25) "
              "-- the largest headroom of any set here. NOTE phantom16, which this repo "
              "previously headlined, is the 3rd-EASIEST of 60 videos (39.1 px median)."),

    # ---------------------------------------------------------------- priority 2: birds
    Dataset(
        key="halmstad", name="Halmstad Drone Detection Dataset", year=2021, modality="rgb-video",
        url="https://github.com/DroneDetectionThesis/Drone-detection-dataset",
        licence="CC0-1.0 (public domain)", gate=Gate.OPEN, verified=True,
        classes=("drone", "bird", "airplane", "helicopter"),
        conditions=(Condition.NIGHT, Condition.CLEAR, Condition.CLUTTER),
        frames=203328, sequences=650, has_birds=True, priority=2,
        notes="The ONLY set that is simultaneously video, bird-labelled, night-inclusive and "
              "CC0 with no form. 365 IR + 285 visible. Friction: MATLAB-format labels "
              "(mcos-decoder), an .xlsx manifest, and NO official split or metric -- you must "
              "define and PUBLISH a split file or the number is not reproducible by anyone."),

    Dataset(
        key="uav_smid", name="UAV_SMID (UAV Sky Monitoring Image Dataset) v2", year=2026,
        modality="rgb-stills", url="https://data.mendeley.com/datasets/3k3hjc7rkt/2",
        licence="CC BY 4.0", gate=Gate.OPEN, verified=True,
        classes=("helicopter", "bomb", "drone", "bird", "aeroplane"),
        conditions=(Condition.CLEAR,), frames=13928, has_birds=True, priority=2,
        notes="13,928 images / 16,229 objects, five DELIBERATELY BALANCED classes "
              "(3,162-3,440 each). Direct download, no form. Stills, so it cannot support "
              "a temporal claim -- its value is as hard negatives: most anti-UAV training "
              "sets are drone-only, so their detectors have never been shown a bird."),

    Dataset(
        key="dvb", name="Drone-vs-Bird (DDS, WOSDETC)", year=2025, modality="rgb-video",
        url="https://github.com/wosdetc/challenge", licence="research-only, signed agreement",
        gate=Gate.AGREEMENT, verified=True,
        classes=("drone",), conditions=(Condition.MOVING_CAMERA, Condition.CLUTTER, Condition.LOW_CONTRAST),
        sequences=77, frames=105000, has_birds=True, priority=5,
        official_protocol=DVB_OFFICIAL,
        notes="The name-recognition benchmark. Birds ARE PRESENT BUT UNLABELLED, so no "
              "bird-attributed false-alarm rate has ever been published in nine editions -- "
              "which is the opening. Requires emailing wosdetc@googlegroups.com and signing "
              "a data-usage agreement; budget a week. Bar to beat: Laroca et al. mAP50 0.7390 "
              "on their 7-video val split."),

    # ---------------------------------------------------------------- priority 3: weather
    Dataset(
        key="extremetrack", name="ExtremeTrack (VISTAC-2, ICPR 2026)", year=2026,
        modality="rgb-video", url="https://sites.google.com/view/vistac-2",
        licence="research", gate=Gate.FORM, verified=True,
        classes=("generic-target",),
        conditions=(Condition.FOG, Condition.RAIN, Condition.LOW_CONTRAST),
        sequences=188, frames=85000, priority=3,
        notes="188 videos = 96 hazy + 92 rainy, splits 128/20/40. The ONLY real "
              "adverse-weather video with per-frame tracking boxes found in the whole sweep. "
              "The ICPR competitions page says 199 videos; the challenge site says 188 and is "
              "authoritative. Submission closed 2026-03-29 -- take the data, not the entry."),

    Dataset(
        key="tricross", name="TriCross-D2D", year=2026, modality="rgb-video",
        url="https://doi.org/10.3390/drones10060459", licence="open access (MDPI)",
        gate=Gate.UNKNOWN, verified=False,
        classes=("uav",),
        conditions=(Condition.FOG, Condition.MOVING_CAMERA, Condition.CLEAR),
        sequences=13, frames=23403, priority=3,
        notes="Air-to-air drone-to-drone with three COUPLED domain shifts: scene, viewpoint "
              "and weather (real flight video plus controlled synthetic fog). 7,045 benchmark "
              "images, 9,771 instances, 73.8% extremely-tiny/tiny/small. The only new set "
              "hitting BOTH stated priorities at once. UNVERIFIED: MDPI returned 403 to "
              "automated fetch, so the download route must be confirmed by hand."),

    # ---------------------------------------------------------------- priority 4: head-to-head
    Dataset(
        key="ard100", name="ARD100", year=2025, modality="rgb-video",
        url="https://github.com/Irisky123/YOLOMG", licence="research", gate=Gate.BAIDU,
        verified=True, classes=("drone",),
        conditions=(Condition.MOVING_CAMERA, Condition.CLUTTER, Condition.NIGHT),
        sequences=100, frames=202467, resolution="1920x1080", priority=4,
        official_protocol=NPS_OFFICIAL,
        notes="YOLOMG's home turf and the tiniest targets of any drone video set "
              "(42.18% under 12x12 px). YOLOMG scores AP 0.85 at 1280 px, 0.78 at 640. "
              "BaiduYun only, no mirror -- start early, expect pain."),

    Dataset(
        key="nps", name="NPS-Drones", year=2019, modality="rgb-video",
        url="https://engineering.purdue.edu/~bouman/UAV_Dataset/", licence="research",
        gate=Gate.UNKNOWN, verified=False, classes=("drone",),
        conditions=(Condition.MOVING_CAMERA, Condition.CLEAR), sequences=50, priority=4,
        official_protocol=NPS_OFFICIAL,
        notes="No official split -- every paper self-splits, so cross-paper NPS numbers are "
              "not comparable. Bar: TransVisDrone and YOLOMG both report AP@0.5 = 0.95."),

    Dataset(
        key="smot4sb", name="MVA 2025 SMOT4SB", year=2025, modality="rgb-video",
        url="https://github.com/IIM-TTIJ/MVA2025-SMOT4SB", licence="research",
        gate=Gate.OPEN, verified=True, classes=("bird",),
        conditions=(Condition.MOVING_CAMERA, Condition.CLUTTER),
        sequences=211, frames=108192, has_birds=True, priority=4,
        notes="108,192 frames of tiny birds from a moving camera and ZERO drones -- the "
              "perfect pure false-positive corpus: 'our detector fires N times across 108k "
              "frames containing no drone' is a number nobody has published. Also supplies "
              "the peer-reviewed cover for centre-distance scoring (SO-HOTA swaps IoU for a "
              "Dot Distance kernel). Live Codabench leaderboard, no end date."),

    Dataset(
        key="dut_antiuav", name="DUT Anti-UAV", year=2022, modality="rgb-stills",
        url="https://github.com/wangdongdut/DUT-Anti-UAV", licence="research",
        gate=Gate.OPEN, verified=True, classes=("uav",),
        conditions=(Condition.CLEAR, Condition.CLUTTER), frames=10000, priority=6,
        notes="The easy end -- SOTA is mAP50 0.92-0.96, meaning the targets are large. "
              "Report it ONLY alongside ARD-MAV/ARD100 with an explicit 'this is the easy "
              "end' sentence, or it invites the comparison that makes 0.83 look bad. Useful "
              "because the Jul-2026 YOLOv11 edge baseline reports on it."),
]}


def by_priority(max_priority: int = 99, video_only: bool = False) -> list[Dataset]:
    ds = [d for d in DATASETS.values() if d.priority <= max_priority]
    if video_only:
        ds = [d for d in ds if d.is_video]
    return sorted(ds, key=lambda d: (d.priority, d.key))


def with_condition(cond: Condition) -> list[Dataset]:
    """Every dataset that can say something about a given operating condition."""
    return sorted((d for d in DATASETS.values() if cond in d.conditions), key=lambda d: d.key)


def with_birds() -> list[Dataset]:
    return sorted((d for d in DATASETS.values() if d.has_birds), key=lambda d: d.priority)
