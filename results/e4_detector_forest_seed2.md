# Step 12 — detecting altered maps (forest_seed2, forest)

Split `test`, seed 0. Surrogate: `geometric_h1`/`gbm`, R² 0.238. Clean residual sd 0.2197.

Each detector is fitted on the first half of the maps and scored on the second.

Paired task: the detector sees each instance's clean and altered map and must separate them. AUC 0.5 is chance. The Mahalanobis baseline is scored at four encoder layers with two poolings each and reported at its best, so it is given every advantage. The map-only column is the control: five statistics of the grid itself, no model involved. A detector that does not beat it has learned nothing about the planner. The last column bootstraps the gap between the combined detector and that control over evaluation instances: an interval straddling zero means the residual added nothing measurable.

| condition | instances | test pairs | residual AUC | residual accuracy | mahalanobis AUC | best layer/pooling | map-only AUC | map-only + residual | gap, 95 % CI |
|-----------|-----------|------------|--------------|-------------------|-----------------|--------------------|--------------|---------------------|--------------|
| `k1_opt` | 100 | 50 | 0.582 | 0.550 | 0.555 | `l5_mean` | 0.768 | 0.753 | -0.014 [-0.052, +0.022] |
| `k1_eff` | 95 | 48 | 0.600 | 0.583 | 0.561 | `l11_mean` | 0.680 | 0.716 | +0.036 [+0.003, +0.081] |

## Against the convolutional map-only control

A small convolutional network given the raw grid, trained on **random** single obstacle-additions to the 800 training maps — never on a selected damaging flip and never on a map it is scored on. This is the hardest model-free bar: it discovers whatever local pattern a flip leaves without being told to look for it, and a defender really could build it, since random flips cost nothing to generate.

It is trained on 1600 examples and detects a random flip on held-out maps with AUC 0.985.

| condition | convolutional control | + residual | gap, 95 % CI |
|-----------|-----------------------|------------|--------------|
| `k1_opt` | 0.917 | 0.898 | -0.018 [-0.053, +0.019] |
| `k1_eff` | 0.907 | 0.914 | +0.007 [-0.030, +0.044] |

## The harder version: no clean twin

Each evaluation map contributes either its clean map or its altered one, never both, so the detector cannot compare a map against itself. This is what a deployed monitor faces, and these are the realistic numbers.

| condition | residual AUC | map-only AUC | map-only + residual | gap, 95 % CI |
|-----------|--------------|--------------|---------------------|--------------|
| `k1_opt` | 0.490 | 0.799 | 0.787 | -0.012 [-0.100, +0.066] |
| `k1_eff` | 0.661 | 0.648 | 0.694 | +0.047 [-0.057, +0.157] |

