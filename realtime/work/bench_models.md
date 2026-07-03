| component | runtime | ms |
|---|---|---|
| verifier n@256 x8 crops | TRT FP16 (5070) | 2.4 |
| full-frame n@1280 | TRT FP16 (5070) | 4.9 |
| full-frame n@640 | TRT FP16 (5070) | 1.7 |
| verifier n@256 x8 crops | ONNX CPU (32-core laptop) | 40.9 |
| full-frame n@1280 | ONNX CPU (32-core laptop) | 153.1 |
| full-frame n@640 | ONNX CPU (32-core laptop) | 43.9 |
| stabilize (768x448 crop corr) | CPU | 3.0 |
| lagged-median motion (amortized) | CPU | 0.2 |
