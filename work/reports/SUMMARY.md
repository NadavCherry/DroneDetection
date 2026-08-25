# SpeckLock -- results

Every number below was produced by this repository: the same videos, splits and labels, the same evaluator, and the same confidence floor (0.001) for every arm. The competitor is **YOLOMG** (arXiv:2503.07115), trained by us under its own published recipe -- 100 epochs at 1280 px against our 30 at 640, i.e. roughly twice our gradient steps.

## ardmav

Protocol `ardmav-official`, split `official-test-15`. Seed-matched paired bootstrap **and** permutation over sequences; significant only when both agree.

| arm | budget | AP mean (per seed) |
|---|---|---|
| **ours** temporal | 30 ep | **0.798** +/- 0.006  (0.805, 0.792, 0.797) |
| **ours** temporal | 100 ep | **0.809** +/- 0.005  (0.806, 0.816, 0.807) |
| ours single-frame (control) | 30 ep | **0.766** +/- 0.006  (0.773, 0.761, 0.763) |
| ours single-frame (control) | 100 ep | **0.768** +/- 0.016  (0.752, 0.784, 0.768) |
| **YOLOMG** (competitor) | 100 ep | **0.834** +/- 0.008  (0.842, 0.828, 0.831) |

### Paired tests, seed-matched

| comparison | seed | d AP | 95% CI | p boot | p perm | verdict |
|---|---|---|---|---|---|---|
| ours temporal - ours single-frame (30 ep) | 0 | +0.032 | [-0.010, +0.087] | 0.1567 | 0.2363 | no difference |
| ours temporal - ours single-frame (30 ep) | 1 | +0.031 | [-0.004, +0.077] | 0.0953 | 0.1699 | no difference |
| ours temporal - ours single-frame (30 ep) | 2 | +0.034 | [-0.019, +0.104] | 0.2500 | 0.3176 | no difference |
| ours temporal - ours single-frame (100 ep) | 0 | +0.054 | [+0.003, +0.120] | 0.0347 | 0.1196 | no difference |
| ours temporal - ours single-frame (100 ep) | 1 | +0.031 | [-0.015, +0.096] | 0.2320 | 0.2979 | no difference |
| ours temporal - ours single-frame (100 ep) | 2 | +0.039 | [-0.011, +0.105] | 0.1500 | 0.2313 | no difference |
| ours temporal 30 ep - YOLOMG | 0 | -0.037 | [-0.079, +0.009] | 0.1107 | 0.1363 | no difference |
| ours temporal 30 ep - YOLOMG | 1 | -0.035 | [-0.069, -0.001] | 0.0447 | 0.0656 | no difference |
| ours temporal 30 ep - YOLOMG | 2 | -0.033 | [-0.074, +0.010] | 0.1353 | 0.1626 | no difference |
| ours temporal 100 ep - YOLOMG | 0 | -0.037 | [-0.079, +0.018] | 0.1693 | 0.1629 | no difference |
| ours temporal 100 ep - YOLOMG | 1 | -0.012 | [-0.048, +0.031] | 0.5973 | 0.5735 | no difference |
| ours temporal 100 ep - YOLOMG | 2 | -0.023 | [-0.055, +0.016] | 0.2260 | 0.2283 | no difference |

### By condition (GLAD's grouping: 5 sequences each)

| arm | ordinary | complex | small |
|---|---|---|---|
| GLAD (published, for placement only) | 0.910 | 0.810 | 0.580 |
| **ours** temporal (30 ep) | 0.926 | 0.804 | 0.651 |
| **ours** temporal (100 ep) | 0.932 | 0.819 | 0.689 |
| ours single-frame (control) (30 ep) | 0.948 | 0.790 | 0.539 |
| ours single-frame (control) (100 ep) | 0.950 | 0.797 | 0.550 |
| **YOLOMG** (competitor) (100 ep) | 0.952 | 0.871 | 0.619 |

#### Paired test on the SMALL condition, ours (100 ep) vs YOLOMG

> Five sequences admit only 2^5 = 32 sign patterns, so the permutation p cannot go below 1/33 = 0.0303 however large the effect. Printed so the floor is not mistaken for strength of evidence.

| seed | d AP | 95% CI | p boot | p perm | verdict |
|---|---|---|---|---|---|
| 0 | +0.071 | [-0.073, +0.168] | 0.2900 | 0.4355 | no difference |
| 1 | +0.060 | [-0.053, +0.153] | 0.2780 | 0.3162 | no difference |
| 2 | +0.079 | [-0.038, +0.173] | 0.1673 | 0.3086 | no difference |

## nps

Protocol `nps-official`, split `nps-no-official-split`. Seed-matched paired bootstrap **and** permutation over sequences; significant only when both agree.

| arm | budget | AP mean (per seed) |
|---|---|---|
| **ours** temporal | 30 ep | **0.480** +/- 0.021  (0.463, 0.503, 0.474) |
| **ours** temporal | 100 ep | **0.487** +/- 0.055  (0.482, 0.544, 0.435) |
| ours single-frame (control) | 30 ep | **0.494** +/- 0.044  (0.544, 0.474, 0.464) |
| ours single-frame (control) | 100 ep | **0.509** +/- 0.022  (0.487, 0.508, 0.532) |
| **YOLOMG** (competitor) | 100 ep | **0.527** +/- 0.027  (0.497, 0.535, 0.548) |

### Paired tests, seed-matched

| comparison | seed | d AP | 95% CI | p boot | p perm | verdict |
|---|---|---|---|---|---|---|
| ours temporal - ours single-frame (30 ep) | 0 | -0.081 | [-0.110, -0.031] | 0.0007 | 0.0203 | **worse** |
| ours temporal - ours single-frame (30 ep) | 1 | +0.029 | [-0.058, +0.105] | 0.5273 | 0.4912 | no difference |
| ours temporal - ours single-frame (30 ep) | 2 | +0.010 | [-0.052, +0.066] | 0.7953 | 0.7824 | no difference |
| ours temporal - ours single-frame (100 ep) | 0 | -0.006 | [-0.050, +0.039] | 0.6387 | 0.9350 | no difference |
| ours temporal - ours single-frame (100 ep) | 1 | +0.036 | [-0.124, +0.117] | 0.6993 | 0.5915 | no difference |
| ours temporal - ours single-frame (100 ep) | 2 | -0.097 | [-0.200, -0.055] | 0.0000 | 0.0227 | **worse** |
| ours temporal 30 ep - YOLOMG | 0 | -0.033 | [-0.128, +0.031] | 0.2340 | 0.6858 | no difference |
| ours temporal 30 ep - YOLOMG | 1 | -0.032 | [-0.107, +0.025] | 0.2527 | 0.3562 | no difference |
| ours temporal 30 ep - YOLOMG | 2 | -0.074 | [-0.131, -0.029] | 0.0000 | 0.2569 | no difference |
| ours temporal 100 ep - YOLOMG | 0 | -0.015 | [-0.146, +0.064] | 0.5307 | 0.7364 | no difference |
| ours temporal 100 ep - YOLOMG | 1 | +0.009 | [-0.116, +0.093] | 0.9927 | 0.8397 | no difference |
| ours temporal 100 ep - YOLOMG | 2 | -0.114 | [-0.196, -0.053] | 0.0000 | 0.0120 | **worse** |

## The project's own videos

One held-out flight, so the interval there is a **moving-block bootstrap over 30-frame blocks WITHIN one sequence**: stability across that flight's segments, not generalisation to another flight. Per-seed tables:

- `local_10_06_seed0.md`
- `local_10_06_seed1.md`
- `local_10_06_seed2.md`
- `local_ft_10_06_seed0.md`
- `local_ft_10_06_seed1.md`
- `local_ft_10_06_seed2.md`
- `local_rev_07_05_seed0.md`
- `local_rev_07_05_seed1.md`
- `local_rev_07_05_seed2.md`

