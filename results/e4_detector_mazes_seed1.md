# Step 12 — detecting altered maps (mazes_seed1, mazes)

Split `test`, seed 0. Surrogate: `geometric_h1`/`gbm`, R² 0.294. Clean residual sd 0.2256.

Paired task: the detector sees each instance's clean and altered map and must separate them. AUC 0.5 is chance. The Mahalanobis baseline is scored at four encoder layers with two poolings each and reported at its best, so it is given every advantage. The map-only column is the control: five statistics of the grid itself, no model involved. A detector that does not beat it has learned nothing about the planner. The last column bootstraps the gap between the combined detector and that control over evaluation instances: an interval straddling zero means the residual added nothing measurable.

| condition | instances | test pairs | residual AUC | residual accuracy | mahalanobis AUC | best layer/pooling | map-only AUC | map-only + residual | gap, 95 % CI |
|-----------|-----------|------------|--------------|-------------------|-----------------|--------------------|--------------|---------------------|--------------|
| `k1_opt` | 100 | 50 | 0.732 | 0.680 | 0.784 | `l5_max` | 0.745 | 0.863 | +0.118 [+0.041, +0.197] |
| `k1_eff` | 100 | 50 | 0.780 | 0.670 | 0.812 | `l5_mean` | 0.631 | 0.882 | +0.252 [+0.171, +0.329] |

