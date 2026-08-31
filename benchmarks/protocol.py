"""Evaluation protocols, as data.

A published AP is meaningless without the protocol that produced it. This project
already had one number quoted beside another that used a different matcher, a different
IoU threshold, a different split and a different dataset -- and nothing in the code could
notice, because the protocol lived in prose.

Here a protocol is an object. `Protocol.mismatches_with` derives the differences between
two of them, so `tools/compare.py` can refuse to print a head-to-head that isn't one.
The rule this encodes: **you may only subtract two numbers that share a protocol.**

Stdlib only, so it runs in the torch-free CI job.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Protocol:
    """How a number was computed. Every field can invalidate a comparison."""

    matcher: str
    """'iou' or 'centre'. A centre-distance AP and an IoU AP are different quantities;
    on a 6 px box they can differ by more than any method ever will (TPAMI 2025 measured
    the same model at IoU-AP 10.9 vs scale-adaptive-AP 24.2)."""

    ap_style: str
    """'ap50' (single IoU 0.5), 'coco' (AP@[.50:.05:.95]), 'voc-all-point' (continuous
    AP at one threshold), or 'ap25' etc. COCO AP and AP50 are not interchangeable."""

    iou_threshold: float | None = None
    """Only for matcher='iou'. ARD-MAV/MGMD use 0.25; most drone papers use 0.5."""

    tau_px: float | None = None
    """Only for matcher='centre'. This repo's default is 12 px."""

    split: str = ""
    """The exact split the number is over, named so it can be checked -- e.g.
    'official-test-15' or 'single-clip-phantom16'. A pooled number and a single-clip
    number are not the same measurement even under an identical matcher."""

    label_inflation_px: float | None = None
    """If training labels were inflated to a fixed minimum side, the size. This bounds
    the achievable IoU and therefore caps any IoU-based score independently of the
    detector -- see docs/research/verified-measurements-2026-08.md §6b."""

    citation: str = ""
    notes: str = ""

    def describe(self) -> str:
        if self.matcher == "iou":
            m = f"IoU≥{self.iou_threshold}" if self.iou_threshold is not None else "IoU"
        else:
            m = f"centre-distance τ={self.tau_px} px"
        parts = [m, self.ap_style]
        if self.split:
            parts.append(f"split={self.split}")
        if self.label_inflation_px:
            parts.append(f"labels inflated to {self.label_inflation_px:g} px")
        return ", ".join(parts)

    def mismatches_with(self, other: "Protocol") -> list[str]:
        """Every reason two numbers under these protocols cannot be subtracted.

        Empty list means the comparison is sound. Anything else must be surfaced in the
        output, not resolved by the caller's judgement.
        """
        out: list[str] = []
        if self.matcher != other.matcher:
            out.append(f"different matcher: {self.describe()} vs {other.describe()}")
        elif self.matcher == "iou" and self.iou_threshold != other.iou_threshold:
            out.append(f"different IoU threshold: {self.iou_threshold} vs {other.iou_threshold}")
        elif self.matcher == "centre" and self.tau_px != other.tau_px:
            out.append(f"different centre-distance radius: τ={self.tau_px} vs τ={other.tau_px} px")

        if self.ap_style != other.ap_style:
            out.append(f"different AP definition: {self.ap_style} vs {other.ap_style}")

        if self.split and other.split and self.split != other.split:
            out.append(f"different split: '{self.split}' vs '{other.split}'")
        elif not self.split or not other.split:
            out.append("split not stated on both sides, so the comparison is unverifiable")

        # An inflated-label model cannot reach a high IoU however well it localises, so
        # an IoU comparison against a true-extent number is capped by geometry.
        if self.matcher == "iou":
            for side, p in (("ours", self), ("theirs", other)):
                if p.label_inflation_px and p.iou_threshold:
                    out.append(
                        f"{side} used {p.label_inflation_px:g} px inflated labels, which caps "
                        f"achievable IoU independently of detector quality")
        return out


# --------------------------------------------------------------------- known protocols

#: What `dronedet` natively produces. Correct at 4 px, and not comparable to any paper.
SPECKLOCK_CENTRE = Protocol(
    matcher="centre", ap_style="voc-all-point", tau_px=12.0,
    citation="dronedet/metrics.py",
    notes="This repo's native rule: match if the detection centre is within "
          "max(tau, 0.5*sqrt(gt area)). Defensible at few-pixel scale and backed by "
          "SAFit (TPAMI 2025) and SO-HOTA (MVA 2025), but it is NOT what papers report. "
          "The citation names metrics.py alone: dronedet/evaluate.py implements the same "
          "radius but resolves a detection that falls inside both a target's and a "
          "distractor's radius the other way, so the two are not interchangeable "
          "implementations of one protocol.")

#: The lingua franca. Produce this for every external claim, whatever else is reported.
COCO = Protocol(
    matcher="iou", ap_style="coco", iou_threshold=0.5,
    citation="COCO detection standard",
    notes="AP averaged over IoU 0.50:0.05:0.95. Brutally pessimistic at 4 px, which is "
          "the point: it is what makes our number and theirs the same kind of number.")

AP50 = Protocol(
    matcher="iou", ap_style="ap50", iou_threshold=0.5,
    citation="the default in most drone-detection papers",
    notes="Single IoU 0.5. What YOLOMG, TransVisDrone and Dogfight report on "
          "NPS-Drones and ARD100.")

#: The bar on ARD-MAV's official split. VERIFIED against the paper 2026-08-12.
#:
#: GLAD, arXiv 2312.11008 / IEEE T-ITS 2024, experiments section, verbatim:
#:     "Following the protocol in [21], the performance evaluation is based on Precision,
#:      Recall, F-Score, and AP. We set the intersection over union (IOU) threshold
#:      between predictions and ground truths to 0.5."
#: on "15 videos from the ARD-MAV dataset", 28,322 frames. The video IDs in `notes` are
#: quoted from the GLAD repository README, which states them explicitly.
ARDMAV_GLAD = Protocol(
    matcher="iou", ap_style="ap50", iou_threshold=0.5, split="official-test-15",
    citation="Guo et al., GLAD, IEEE T-ITS 2024 (arXiv 2312.11008)",
    notes="15 held-out videos: phantom{05,08,09,10,19,30,41,43,46,47,58,63,65,70,86}, "
          "28,322 frames. GLAD scores AP 0.80 overall (P 0.92 / R 0.82), and by its own "
          "three conditions: ordinary 0.91, complex background 0.81, SMALL MAVs 0.58. "
          "The small-MAV row is the one this project is built to contest.")

#: GLAD's per-condition rows. Same threshold and AP style as ARDMAV_GLAD, DIFFERENT split:
#: each is 5 of the 15 videos, so a per-condition number and an overall number are not on
#: the same axis and `mismatches_with` must say so.
#:
#: This exists because the first comparison run got it wrong in our favour. With the
#: small-MAV row carrying split="official-test-15", the table compared our OVERALL AP
#: (0.754, all 15 videos) against GLAD's SMALL-subset AP (0.58, 5 videos) and printed
#: "ours higher". Our actual small-subset AP is 0.530, which loses. Same protocol object,
#: different populations -- the one kind of mismatch a shared `split` string hides.
ARDMAV_GLAD_ORDINARY = replace(ARDMAV_GLAD, split="official-test-15/ordinary",
                               notes="phantom09,10,30,47,70. GLAD 0.91.")
ARDMAV_GLAD_COMPLEX = replace(ARDMAV_GLAD, split="official-test-15/complex",
                              notes="phantom05,08,58,65,86. GLAD 0.81.")
ARDMAV_GLAD_SMALL = replace(ARDMAV_GLAD, split="official-test-15/small",
                            notes="phantom19,41,43,46,63. GLAD 0.58 -- its weakest "
                                  "condition and the one this project is built to win.")

#: NOT the official split. MGMD scores at IoU 0.25 on a split of its own that it never
#: enumerates, so a number under this protocol is not placeable beside a GLAD number and
#: `mismatches_with` will say so on both counts (threshold AND split).
ARDMAV_MGMD = Protocol(
    matcher="iou", ap_style="ap25", iou_threshold=0.25,
    split="mgmd-self-chosen-UNENUMERATED",
    citation="MGMD (ARD-MAV authors), IoU 0.25 on an unpublished split",
    notes="Kept only so a quoted MGMD number can be typed without pretending it is "
          "comparable. The split is not published, so nobody outside that lab can "
          "reproduce it and nobody can score against it.")

#: BACKWARDS-COMPATIBILITY ALIAS, and a correction.
#:
#: `ARDMAV_OFFICIAL` used to be `iou_threshold=0.25, split="official-test-15"`, citing
#: "Guo et al., GLAD / MGMD" -- one Protocol object splicing two papers, binding MGMD's
#: threshold to GLAD's split. That is precisely the error this module exists to prevent,
#: and it was compiled into the engine that decides what is comparable: every ARD-MAV
#: number this repo produced was being scored at a threshold matching NO published number
#: on that split, and the bar was recorded as 0.55 when the bar is 0.80 at a threshold
#: twice as strict. Now points at the verified GLAD protocol.
ARDMAV_OFFICIAL = ARDMAV_GLAD

NPS_OFFICIAL = Protocol(
    matcher="iou", ap_style="ap50", iou_threshold=0.5, split="nps-no-official-split",
    citation="TransVisDrone / YOLOMG report AP@0.5 on NPS-Drones",
    notes="NPS-Drones publishes no official split, so every paper self-splits. Say so "
          "whenever an NPS number is quoted rather than implying a standard.")

DVB_OFFICIAL = Protocol(
    matcher="iou", ap_style="ap50", iou_threshold=0.5, split="dvb-self-chosen-val",
    citation="WOSDETC Drone-vs-Bird Grand Challenge",
    notes="Test annotations are withheld, so EVERY published DvB number is on a "
          "self-chosen validation split. Three such numbers side by side are three "
          "different experiments.")

BY_KEY: dict[str, Protocol] = {
    "specklock-centre": SPECKLOCK_CENTRE,
    "coco": COCO,
    "ap50": AP50,
    "ardmav-official": ARDMAV_OFFICIAL,
    "nps-official": NPS_OFFICIAL,
    "dvb-official": DVB_OFFICIAL,
}
