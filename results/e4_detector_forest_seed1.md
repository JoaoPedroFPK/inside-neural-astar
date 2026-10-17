# Step 12 — detecting altered maps (forest_seed1, forest)

Split `test`, seed 0. Surrogate: `geometric_h1`/`gbm`, R² 0.301. Clean residual sd 0.1971.

Each detector is fitted on the first half of the maps and scored on the second.

Paired task: the detector sees each instance's clean and altered map and must separate them. AUC 0.5 is chance. The Mahalanobis baseline is scored at four encoder layers with two poolings each and reported at its best, so it is given every advantage. The map-only column is the control: five statistics of the grid itself, no model involved. A detector that does not beat it has learned nothing about the planner. The last column bootstraps the gap between the combined detector and that control over evaluation instances: an interval straddling zero means the residual added nothing measurable.

| condition | instances | test pairs | residual AUC | residual accuracy | mahalanobis AUC | best layer/pooling | map-only AUC | map-only + residual | gap, 95 % CI |
|-----------|-----------|------------|--------------|-------------------|-----------------|--------------------|--------------|---------------------|--------------|
| `k1_opt` | 100 | 50 | 0.607 | 0.580 | 0.595 | `l8_mean` | 0.855 | 0.848 | -0.007 [-0.041, +0.021] |
| `k1_eff` | 93 | 47 | 0.595 | 0.543 | 0.567 | `l2_max` | 0.709 | 0.718 | +0.009 [-0.026, +0.049] |

