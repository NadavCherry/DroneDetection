# Edge model: accuracy against speed

GPU: **NVIDIA GeForce RTX 4090**. Video: `10_06.mp4`, scored against `gt_1006_v2.json` (rule=centre, tau=12.0). Speed and accuracy come from the SAME pass, so they describe one execution and cannot drift apart.

The published 74 / 84.8 / 104 fps figures were taken on an RTX 5070 Laptop and are **not** comparable to these.

| arm | backend | imgsz | AP | recall | precision | **fps (steady p50)** | p50 ms | p95 ms | p99 ms | end-to-end fps | warm-up cost |
|---|---|---|---|---|---|---|---|---|---|---|---|
| edge-pt-1280 | pt | 1280 | **0.8793** | 0.875 | 0.799 | **35.2** | 28.407 | 30.512 | 32.067 | 29.04 | 1.5 s |
| edge-pt-640 | pt | 640 | **0.6464** | 0.605 | 0.891 | **40.2** | 24.9 | 26.784 | 27.552 | 34.56 | 0.7 s |
| edge-engine-1280 | engine | 1280 | **0.8757** | 0.858 | 0.812 | **58.9** | 16.979 | 18.853 | 19.922 | 30.43 | 4.9 s |
| edge-engine-640 | engine | 640 | **0.6393** | 0.602 | 0.898 | **72.1** | 13.869 | 15.265 | 15.799 | 57.13 | 0.5 s |

Per-stage means (ms/frame):

| arm | TOTAL ms/frame | detector NN | stabilize+warp |
|---|---|---|---|
| edge-pt-1280 | 29.196 | 15.307 | 13.889 |
| edge-pt-640 | 24.02 | 10.206 | 13.813 |
| edge-engine-1280 | 28.349 | 20.2 | 8.149 |
| edge-engine-640 | 13.139 | 5.158 | 7.981 |

