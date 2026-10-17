# Step 12 — detecting altered maps (mazes_seed0, mazes)

Split `test`, seed 0. Surrogate: `geometric_h1`/`gbm`, R² 0.274. Clean residual sd 0.2265.

Each detector is fitted on the first half of the maps and scored on the second.

Paired task: the detector sees each instance's clean and altered map and must separate them. AUC 0.5 is chance. The Mahalanobis baseline is scored at four encoder layers with two poolings each and reported at its best, so it is given every advantage. The map-only column is the control: five statistics of the grid itself, no model involved. A detector that does not beat it has learned nothing about the planner. The last column bootstraps the gap between the combined detector and that control over evaluation instances: an interval straddling zero means the residual added nothing measurable.

| condition | instances | test pairs | residual AUC | residual accuracy | mahalanobis AUC | best layer/pooling | map-only AUC | map-only + residual | gap, 95 % CI |
|-----------|-----------|------------|--------------|-------------------|-----------------|--------------------|--------------|---------------------|--------------|
| `k1_opt` | 100 | 50 | 0.710 | 0.620 | 0.816 | `l2_mean` | 0.736 | 0.874 | +0.137 [+0.068, +0.220] |
| `k1_eff` | 100 | 50 | 0.767 | 0.740 | 0.818 | `l5_mean` | 0.606 | 0.854 | +0.248 [+0.178, +0.321] |

## Against the convolutional map-only control

A small convolutional network given the raw grid, trained on **random** single obstacle-additions to the 800 training maps — never on a selected damaging flip and never on a map it is scored on. This is the hardest model-free bar: it discovers whatever local pattern a flip leaves without being told to look for it, and a defender really could build it, since random flips cost nothing to generate.

It is trained on 1600 examples and detects a random flip on held-out maps with AUC 1.000.

| condition | convolutional control | + residual | gap, 95 % CI |
|-----------|-----------------------|------------|--------------|
| `k1_opt` | 0.914 | 0.944 | +0.029 [-0.011, +0.078] |
| `k1_eff` | 0.817 | 0.897 | +0.080 [+0.019, +0.142] |

## The harder version: no clean twin

Each evaluation map contributes either its clean map or its altered one, never both, so the detector cannot compare a map against itself. This is what a deployed monitor faces, and these are the realistic numbers.

| condition | residual AUC | map-only AUC | map-only + residual | gap, 95 % CI |
|-----------|--------------|--------------|---------------------|--------------|
| `k1_opt` | 0.640 | 0.669 | 0.880 | +0.211 [+0.089, +0.356] |
| `k1_eff` | 0.739 | 0.554 | 0.840 | +0.286 [+0.093, +0.483] |

