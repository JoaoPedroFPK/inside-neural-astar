# Step 11 — WarCraft 12×12

Models: `warcraft_seed0`, `warcraft_seed1`, `warcraft_seed2`. Ranges are min–max across models.

## Baseline (test split, 1 000 maps)

| quantity | warcraft_seed0 | warcraft_seed1 | warcraft_seed2 | range |
|---|---|---|---|---|
| Opt binary (planner) | 0.374 | 0.329 | 0.368 | 0.329–0.374 |
| Opt mean ratio | 1.118 | 1.145 | 1.137 | 1.118–1.145 |
| Exp mean ratio | 0.581 | 0.493 | 0.498 | 0.493–0.581 |
| Opt binary, exact search on learned costs | 0.320 | 0.321 | 0.318 | 0.318–0.321 |
| Opt binary, classical A* on true weights | 0.924 | 0.924 | 0.924 | 0.924 |
| Spearman(cost, true weight), per-map mean | 0.317 | 0.398 | 0.332 | 0.317–0.398 |
| learned scale `const` | 11.693 | 11.529 | 11.695 | 11.529–11.695 |

Mean learned cost per terrain class:

| true cost | warcraft_seed0 | warcraft_seed1 | warcraft_seed2 |
|---|---|---|---|
| 0.8 | 1.38 ± 1.47 | 1.24 ± 1.53 | 1.26 ± 1.38 |
| 1.2 | 1.57 ± 1.32 | 1.46 ± 1.36 | 1.45 ± 1.23 |
| 5.3 | 2.19 ± 1.62 | 1.97 ± 1.63 | 1.97 ± 1.55 |
| 7.7 | 2.05 ± 1.59 | 2.16 ± 1.71 | 2.02 ± 1.59 |
| 9.2 | 1.84 ± 1.46 | 1.85 ± 1.49 | 1.70 ± 1.43 |

## Surrogate fits (R² on test, cost map from tile features)

| group | model | R² range | within-map R² range |
|---|---|---|---|
| geometric | ridge | 0.056–0.068 | 0.057–0.069 |
| terrain | ridge | 0.041–0.046 | 0.038–0.039 |
| terrain | gbm | 0.089–0.100 | 0.085–0.092 |
| path | ridge | 0.037–0.044 | 0.036–0.041 |
| path | gbm | 0.078–0.086 | 0.076–0.081 |
| terrain+geometric | ridge | 0.094–0.106 | 0.091–0.101 |
| terrain+path | ridge | 0.088–0.097 | 0.086–0.092 |
| all | ridge | 0.101–0.114 | 0.099–0.110 |
| all | gbm | 0.846–0.875 | 0.847–0.876 |

Sufficiency — the same search run on a substitute cost map:

| cost map | Opt binary range | Exp mean range |
|---|---|---|
| terrain/ridge | 0.072–0.078 | 1.225–1.406 |
| terrain/gbm | 0.186–0.192 | 1.203–1.424 |
| path/gbm | 0.012–0.031 | 1.905–2.080 |
| all/ridge | 0.000–0.018 | 2.520–2.874 |
| all/gbm | 0.372–0.431 | 0.630–1.032 |
| true_weights | 0.924 | 1.000 |

## Random tile transplants (all 1 000 test maps, 3 draws per budget)

| k | suboptimal | worsened | Exp > 1.5× own baseline | path changed |
|---|---|---|---|---|
| 1 | 0.657–0.696 | 0.140–0.161 | 0.008–0.010 | 0.188–0.212 |
| 2 | 0.690–0.720 | 0.255–0.282 | 0.015–0.021 | 0.338–0.377 |
| 4 | 0.724–0.766 | 0.408–0.422 | 0.025–0.042 | 0.514–0.566 |
| 8 | 0.821–0.838 | 0.562–0.608 | 0.048–0.069 | 0.720–0.776 |

## Exhaustive single-tile transplants (300 test maps, 170400 edits per model)

| quantity | range |
|---|---|
| baseline-optimal maps | 99–107 |
| … with ≥ 1 damaging edit | 1.000 |
| median share of damaging edits per map | 0.118–0.134 |
| share of edits that need a response (old path no longer optimal) | 0.673–0.698 |
| share of edits that change the planner's path | 0.187–0.210 |
| worst Opt ratio | 3.206–4.412 |
| Spearman(ΔCost L1, Opt damage) | 0.104–0.129 |
| share of ΔCost on the edited tile | 0.138–0.142 |
| max ΔCost beyond the receptive field | 1.6e-04 |

Damage by location of the edited tile (baseline-optimal maps):

| location | share damaging | costlier edit | cheaper edit |
|---|---|---|---|
| away from optimal paths | 0.039–0.047 | 0.026–0.031 | 0.057–0.069 |
| next to an optimal path | 0.177–0.195 | 0.132–0.143 | 0.252–0.287 |
| on another optimal path | 0.306–0.323 | 0.195–0.227 | 0.639–0.653 |
| on planner path | 0.552–0.595 | 0.649–0.683 | 0.144–0.201 |

What carries the damage (damaging edits on baseline-optimal maps):

| quantity | range |
|---|---|
| damaging edits | 7762–9299 |
| cost-ranking (exact search on the altered costs also misses the optimum) | 0.820–0.850 |
| search-rule (exact search on the altered costs finds it) | 0.150–0.180 |
| unprovoked (the old path was still optimal) | 0.506–0.566 |

## L∞ pixel perturbations (true costs unchanged)

300 test maps; PGD with 10 steps on upstream's training loss.

| ε (×1/255) | method | Opt binary | optimal maps broken | worsened | path changed |
|---|---|---|---|---|---|
| 1 | random_sign | 0.313–0.367 | 0.038–0.081 | 0.040–0.063 | 0.130–0.180 |
| 1 | pgd | 0.053–0.057 | 0.838–0.850 | 0.640–0.660 | 0.810–0.820 |
| 2 | random_sign | 0.330–0.367 | 0.104–0.112 | 0.083–0.113 | 0.207–0.307 |
| 2 | pgd | 0.033–0.043 | 0.869–0.934 | 0.753–0.807 | 0.903–0.927 |
| 4 | random_sign | 0.320–0.363 | 0.152–0.243 | 0.130–0.237 | 0.420–0.497 |
| 4 | pgd | 0.017–0.037 | 0.899–0.972 | 0.817–0.877 | 0.947–0.977 |
| 8 | random_sign | 0.240–0.340 | 0.313–0.514 | 0.243–0.403 | 0.617–0.730 |
| 8 | pgd | 0.013–0.040 | 0.899–0.972 | 0.867–0.920 | 0.960–0.993 |
| 16 | random_sign | 0.117–0.177 | 0.657–0.832 | 0.473–0.657 | 0.880–0.920 |
| 16 | pgd | 0.010–0.037 | 0.899–0.981 | 0.897–0.930 | 0.967–0.990 |
