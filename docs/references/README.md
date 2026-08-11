# Reference material

The research surveys that drove this project's design — collected here so the whole "why" is in one place. The evidence-backed impact ranking from these sources is what set our priority order: **motion/temporal cues > resolution preservation > tiling (SAHI) > tiny-aware loss (NWD/RFLA) > P2 head**, evaluated with size-binned **center-distance** metrics rather than mAP@0.5.

| file | what it is |
|---|---|
| [tiny-drone-detection-survey.md](tiny-drone-detection-survey.md) | *Detection and Localization of Tiny Drones and Sub-pixel Objects in Images and Video* — deep architectures, super-resolution, and integration of classical algorithms. (One long single line of text; line-based tools are awkward on it.) |
| [few-pixel-drone-methods-survey.md](few-pixel-drone-methods-survey.md) | *Detecting Few-Pixel Drones in Camera Video: A Technical Survey of Best-Quality Methods* — the motion-fusion + resolution-preservation playbook, with the evidence-backed impact rankings we followed. |
| [tiny-object-detection-hebrew.docx](tiny-object-detection-hebrew.docx) | Hebrew-language source document on the same topic. (3.0 MB, and it carries no document properties — no author, no date.) |

**Provenance.** None of the three is a published paper: they are working literature syntheses
gathered while scoping this project, they carry no byline, venue or date, and they were never
peer-reviewed. Read them as a bibliography with commentary. Every quantitative claim inside them is
attributed inline to the paper it comes from, and the quoted passages belong to those authors — cite
the original, not this file. What is *measured here*, on this repo's own video, is in the
[reports](../reports/); where the two disagree, the reports win.

How the design maps onto these findings is documented throughout the [reports](../reports/) and the [methods guide](../guides/methods.md).
