# Step 12 — detecting altered maps (mazes_seed2, mazes)

Split `test`, seed 0. Surrogate: `geometric_h1`/`gbm`, R² 0.308. Clean residual sd 0.2295.

Each detector is fitted on the first half of the maps and scored on the second.

Paired task: the detector sees each instance's clean and altered map and must separate them. AUC 0.5 is chance. The Mahalanobis baseline is scored at four encoder layers with two poolings each and reported at its best, so it is given every advantage. The map-only column is the control: five statistics of the grid itself, no model involved. A detector that does not beat it has learned nothing about the planner. The last column bootstraps the gap between the combined detector and that control over evaluation instances: an interval straddling zero means the residual added nothing measurable.

| condition | instances | test pairs | residual AUC | residual accuracy | mahalanobis AUC | best layer/pooling | map-only AUC | map-only + residual | gap, 95 % CI |
|-----------|-----------|------------|--------------|-------------------|-----------------|--------------------|--------------|---------------------|--------------|
| `k1_opt` | 100 | 50 | 0.665 | 0.620 | 0.787 | `l2_mean` | 0.750 | 0.862 | +0.111 [+0.048, +0.176] |
| `k1_eff` | 100 | 50 | 0.772 | 0.720 | 0.788 | `l2_mean` | 0.639 | 0.867 | +0.228 [+0.146, +0.298] |

## The harder version: no clean twin

Each evaluation map contributes either its clean map or its altered one, never both, so the detector cannot compare a map against itself. This is what a deployed monitor faces, and these are the realistic numbers.

| condition | residual AUC | map-only AUC | map-only + residual | gap, 95 % CI |
|-----------|--------------|--------------|---------------------|--------------|
| `k1_opt` | 0.770 | 0.691 | 0.882 | +0.190 [+0.035, +0.356] |
| `k1_eff` | 0.864 | 0.572 | 0.941 | +0.369 [+0.195, +0.548] |

