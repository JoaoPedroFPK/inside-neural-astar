# Step 12 — detecting altered maps (pretrained, mazes)

Split `test`, seed 0. Surrogate: `geometric_h1`/`gbm`, R² 0.240. Clean residual sd 0.2434.

Detector trained on `greedy_opt_b4` and scored on every condition below without refitting — the transfer test `PLAN.md` asks for.

Paired task: the detector sees each instance's clean and altered map and must separate them. AUC 0.5 is chance. The Mahalanobis baseline is scored at four encoder layers with two poolings each and reported at its best, so it is given every advantage. The map-only column is the control: five statistics of the grid itself, no model involved. A detector that does not beat it has learned nothing about the planner. The last column bootstraps the gap between the combined detector and that control over evaluation instances: an interval straddling zero means the residual added nothing measurable.

| condition | instances | test pairs | residual AUC | residual accuracy | mahalanobis AUC | best layer/pooling | map-only AUC | map-only + residual | gap, 95 % CI |
|-----------|-----------|------------|--------------|-------------------|-----------------|--------------------|--------------|---------------------|--------------|
| `k1_opt` | 100 | 50 | 0.654 | 0.590 | 0.841 | `l2_mean` | 0.772 | 0.897 | +0.125 [+0.061, +0.183] |
| `k1_eff` | 100 | 50 | 0.773 | 0.670 | 0.833 | `l2_mean` | 0.623 | 0.803 | +0.180 [+0.112, +0.250] |
| `greedy_opt_b1` | 100 | 50 | 0.654 | 0.590 | 0.841 | `l2_mean` | 0.772 | 0.897 | +0.125 [+0.061, +0.183] |
| `greedy_opt_b4` | 100 | 50 | 0.798 | 0.700 | 0.978 | `l2_mean` | 0.910 | 0.920 | +0.010 [-0.030, +0.049] |
| `greedy_opt_b8` | 100 | 50 | 0.817 | 0.720 | 0.980 | `l2_mean` | 0.904 | 0.932 | +0.028 [-0.006, +0.066] |
| `greedy_eff_b1` | 94 | 49 | 0.776 | 0.673 | 0.834 | `l2_mean` | 0.626 | 0.805 | +0.180 [+0.112, +0.251] |
| `greedy_eff_b4` | 94 | 49 | 0.894 | 0.765 | 0.979 | `l2_mean` | 0.946 | 0.970 | +0.024 [-0.008, +0.063] |
| `greedy_eff_b8` | 94 | 49 | 0.952 | 0.816 | 0.993 | `l8_mean` | 0.939 | 0.990 | +0.051 [+0.006, +0.103] |
| `balanced_2` | 100 | 50 | 0.778 | 0.670 | 0.952 | `l2_mean` | 0.786 | 0.920 | +0.134 [+0.073, +0.192] |
| `blocked_2` | 100 | 50 | 0.688 | 0.600 | 0.955 | `l2_mean` | 0.932 | 0.976 | +0.045 [+0.005, +0.090] |

## The harder version: no clean twin

Each evaluation map contributes either its clean map or its altered one, never both, so the detector cannot compare a map against itself. This is what a deployed monitor faces, and these are the realistic numbers.

| condition | residual AUC | map-only AUC | map-only + residual | gap, 95 % CI |
|-----------|--------------|--------------|---------------------|--------------|
| `k1_opt` | 0.658 | 0.719 | 0.912 | +0.193 [+0.061, +0.345] |
| `k1_eff` | 0.790 | 0.545 | 0.781 | +0.236 [+0.080, +0.399] |
| `greedy_opt_b1` | 0.658 | 0.719 | 0.912 | +0.193 [+0.061, +0.345] |
| `greedy_opt_b4` | 0.818 | 0.893 | 0.931 | +0.038 [-0.026, +0.114] |
| `greedy_opt_b8` | 0.834 | 0.890 | 0.938 | +0.048 [-0.018, +0.125] |
| `greedy_eff_b1` | 0.795 | 0.470 | 0.738 | +0.268 [+0.102, +0.444] |
| `greedy_eff_b4` | 0.948 | 0.960 | 0.993 | +0.033 [+0.000, +0.095] |
| `greedy_eff_b8` | 0.982 | 0.897 | 1.000 | +0.103 [+0.010, +0.215] |
| `balanced_2` | 0.811 | 0.743 | 0.931 | +0.188 [+0.078, +0.325] |
| `blocked_2` | 0.712 | 0.921 | 0.970 | +0.049 [+0.000, +0.126] |

