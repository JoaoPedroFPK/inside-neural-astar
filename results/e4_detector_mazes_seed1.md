# Step 12 — detecting altered maps (mazes_seed1, mazes)

Split `test`, seed 0. Surrogate: `geometric_h1`/`gbm`, R² 0.294. Clean residual sd 0.2256.

Each detector is fitted on the first half of the maps and scored on the second.

Paired task: the detector sees each instance's clean and altered map and must separate them. AUC 0.5 is chance. The Mahalanobis baseline is scored at four encoder layers with two poolings each and reported at its best, so it is given every advantage. The map-only column is the control: five statistics of the grid itself, no model involved. A detector that does not beat it has learned nothing about the planner. The last column bootstraps the gap between the combined detector and that control over evaluation instances: an interval straddling zero means the residual added nothing measurable.

| condition | instances | test pairs | residual AUC | residual accuracy | mahalanobis AUC | best layer/pooling | map-only AUC | map-only + residual | gap, 95 % CI |
|-----------|-----------|------------|--------------|-------------------|-----------------|--------------------|--------------|---------------------|--------------|
| `k1_opt` | 100 | 50 | 0.732 | 0.680 | 0.784 | `l5_max` | 0.745 | 0.863 | +0.118 [+0.041, +0.197] |
| `k1_eff` | 100 | 50 | 0.780 | 0.670 | 0.812 | `l5_mean` | 0.631 | 0.882 | +0.252 [+0.171, +0.329] |

## Against the convolutional map-only control

A small convolutional network given the raw grid, trained on **random** single obstacle-additions to the 800 training maps — never on a selected damaging flip and never on a map it is scored on. This is the hardest model-free bar: it discovers whatever local pattern a flip leaves without being told to look for it, and a defender really could build it, since random flips cost nothing to generate.

It is trained on 1600 examples and detects a random flip on held-out maps with AUC 1.000.

| condition | convolutional control | + residual | gap, 95 % CI |
|-----------|-----------------------|------------|--------------|
| `k1_opt` | 0.926 | 0.974 | +0.048 [+0.004, +0.100] |
| `k1_eff` | 0.788 | 0.903 | +0.115 [+0.048, +0.179] |

## The harder version: no clean twin

Each evaluation map contributes either its clean map or its altered one, never both, so the detector cannot compare a map against itself. This is what a deployed monitor faces, and these are the realistic numbers.

| condition | residual AUC | map-only AUC | map-only + residual | gap, 95 % CI |
|-----------|--------------|--------------|---------------------|--------------|
| `k1_opt` | 0.693 | 0.716 | 0.870 | +0.154 [+0.002, +0.317] |
| `k1_eff` | 0.787 | 0.610 | 0.917 | +0.306 [+0.131, +0.489] |

