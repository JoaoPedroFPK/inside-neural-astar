# Step 12 — detecting altered maps (forest_seed0, forest)

Split `test`, seed 0. Surrogate: `geometric_h1`/`gbm`, R² 0.205. Clean residual sd 0.2045.

Paired task: the detector sees each instance's clean and altered map and must separate them. AUC 0.5 is chance. The Mahalanobis baseline is scored at four encoder layers with two poolings each and reported at its best, so it is given every advantage. The map-only column is the control: five statistics of the grid itself, no model involved. A detector that does not beat it has learned nothing about the planner. The last column bootstraps the gap between the combined detector and that control over evaluation instances: an interval straddling zero means the residual added nothing measurable.

| condition | instances | test pairs | residual AUC | residual accuracy | mahalanobis AUC | best layer/pooling | map-only AUC | map-only + residual | gap, 95 % CI |
|-----------|-----------|------------|--------------|-------------------|-----------------|--------------------|--------------|---------------------|--------------|
| `k1_opt` | 100 | 50 | 0.582 | 0.560 | 0.584 | `l5_mean` | 0.743 | 0.742 | -0.001 [-0.036, +0.036] |
| `k1_eff` | 95 | 48 | 0.539 | 0.552 | 0.553 | `l8_mean` | 0.696 | 0.650 | -0.046 [-0.098, +0.004] |

