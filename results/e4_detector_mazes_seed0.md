# Step 12 — detecting altered maps (mazes_seed0, mazes)

Split `test`, seed 0. Surrogate: `geometric_h1`/`gbm`, R² 0.274. Clean residual sd 0.2265.

Each detector is fitted on the first half of the maps and scored on the second.

Paired task: the detector sees each instance's clean and altered map and must separate them. AUC 0.5 is chance. The Mahalanobis baseline is scored at four encoder layers with two poolings each and reported at its best, so it is given every advantage. The map-only column is the control: five statistics of the grid itself, no model involved. A detector that does not beat it has learned nothing about the planner. The last column bootstraps the gap between the combined detector and that control over evaluation instances: an interval straddling zero means the residual added nothing measurable.

| condition | instances | test pairs | residual AUC | residual accuracy | mahalanobis AUC | best layer/pooling | map-only AUC | map-only + residual | gap, 95 % CI |
|-----------|-----------|------------|--------------|-------------------|-----------------|--------------------|--------------|---------------------|--------------|
| `k1_opt` | 100 | 50 | 0.710 | 0.620 | 0.816 | `l2_mean` | 0.736 | 0.874 | +0.137 [+0.068, +0.220] |
| `k1_eff` | 100 | 50 | 0.767 | 0.740 | 0.818 | `l5_mean` | 0.606 | 0.854 | +0.248 [+0.178, +0.321] |

## The harder version: no clean twin

Each evaluation map contributes either its clean map or its altered one, never both, so the detector cannot compare a map against itself. This is what a deployed monitor faces, and these are the realistic numbers.

| condition | residual AUC | map-only AUC | map-only + residual | gap, 95 % CI |
|-----------|--------------|--------------|---------------------|--------------|
| `k1_opt` | 0.640 | 0.669 | 0.880 | +0.211 [+0.089, +0.356] |
| `k1_eff` | 0.739 | 0.554 | 0.840 | +0.286 [+0.093, +0.483] |

