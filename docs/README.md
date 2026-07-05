# Documentation

Everything that isn't code. Start at the [project README](../README.md); come here to go deep.

## Guides — how to use the project

| guide | for |
|---|---|
| [guides/methods.md](guides/methods.md) | **understanding** — every algorithm's full pipeline, the models inside it, and its measured performance (+ the weights map) |
| [guides/run-inference.md](guides/run-inference.md) | **running** our models (or the baseline) on a new video |
| [guides/retrain.md](guides/retrain.md) | **retraining** — relabel a clip, what the dataset consists of, build datasets, train, reproduce a round |

Deliverable-specific docs live next to the code: [final/README.md](../final/README.md) (the two shipped models) and [realtime/README.md](../realtime/README.md) (the edge pipeline in depth).

## Reports — the three-round build story

The design narrative, in order. Each reads on its own; together they are how the pipeline was earned.

| report | round |
|---|---|
| [reports/round1-pipeline.md](reports/round1-pipeline.md) | 1 — building the pipeline; the resolution/SAHI/motion comparisons; the tiny-object training failure and its fix |
| [reports/round2-results.md](reports/round2-results.md) | 2 — evaluation against hand labels; the temporal-vs-single-frame ablation; the slow-mover fix; SR & bird handling |
| [reports/round3-deliverables.md](reports/round3-deliverables.md) | 3 — track-level classification; v3 data; the dense test reference; the two shipped models |
| [reports/round4-external-datasets.md](reports/round4-external-datasets.md) | 4 — public tiny-drone datasets (ARD-MAV train, NPS unseen test); train/val/test + cross-dataset generalization; comparison vs GLAD / YOLOMG / Dogfight / TransVisDrone; motion-guided improvement plan |
| [reports/round5-moving-camera-multidataset.md](reports/round5-moving-camera-multidataset.md) | 5 — moving camera + one combined dataset (per-dataset video splits); strongest PC detector (colour+scale aug); new ego-motion-compensated motion front-end; method comparison; full pipeline tracks the black drone at 99.7% |
| [reports/round6-max-pipeline.md](reports/round6-max-pipeline.md) | 6 — unified MAX pipeline (regime-adaptive fusion → affine tracker → track-level classification) in one command; temporal motion-in-input expert; v1 vs v2 — combining wins at detection (black drone 0.52→0.69) but tracking saturates; drone-vs-bird is the frontier |

## References — the research that drove the design

See [references/README.md](references/README.md). The tiny-object-detection surveys behind the design choices (motion-first pipelines, resolution preservation, SAHI tiling, tiny-aware loss, center-distance evaluation).

## Media

[media/](media/) holds the showcase videos, GIFs, and the architecture SVGs used across the READMEs. Regenerate the architecture figures with [`tools/make_arch_figures_final.py`](../tools/make_arch_figures_final.py) and [`tools/make_arch_figure.py`](../tools/make_arch_figure.py).
