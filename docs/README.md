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

## References — the research that drove the design

See [references/README.md](references/README.md). The tiny-object-detection surveys behind the design choices (motion-first pipelines, resolution preservation, SAHI tiling, tiny-aware loss, center-distance evaluation).

## Media

[media/](media/) holds the showcase videos, GIFs, and the architecture SVGs used across the READMEs. Regenerate the architecture figures with [`tools/make_arch_figures_final.py`](../tools/make_arch_figures_final.py) and [`tools/make_arch_figure.py`](../tools/make_arch_figure.py).
