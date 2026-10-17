# Step 12 — detecting altered maps (mazes_seed2, mazes)

Split `test`, seed 0. Surrogate: `geometric_h1`/`gbm`, R² 0.308. Clean residual sd 0.2295.

Paired task: the detector sees each instance's clean and altered map and must separate them. AUC 0.5 is chance. The Mahalanobis baseline is scored at four encoder layers with two poolings each and reported at its best, so it is given every advantage. The map-only column is the control: five statistics of the grid itself, no model involved. A detector that does not beat it has learned nothing about the planner. The last column bootstraps the gap between the combined detector and that control over evaluation instances: an interval straddling zero means the residual added nothing measurable.

| condition | instances | test pairs | residual AUC | residual accuracy | mahalanobis AUC | best layer/pooling | map-only AUC | map-only + residual | gap, 95 % CI |
|-----------|-----------|------------|--------------|-------------------|-----------------|--------------------|--------------|---------------------|--------------|
| `k1_opt` | 100 | 50 | 0.665 | 0.620 | 0.787 | `l2_mean` | 0.750 | 0.862 | +0.111 [+0.048, +0.176] |
| `k1_eff` | 100 | 50 | 0.772 | 0.720 | 0.788 | `l2_mean` | 0.639 | 0.867 | +0.228 [+0.146, +0.298] |

