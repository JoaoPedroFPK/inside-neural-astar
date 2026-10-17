# Step 12 — detecting altered maps (pretrained, mazes)

Split `test`, seed 0. Surrogate: `geometric_h1`/`gbm`, R² 0.240. Clean residual sd 0.2434.

Each detector is fitted on the first half of the maps and scored on the second.

Paired task: the detector sees each instance's clean and altered map and must separate them. AUC 0.5 is chance. The Mahalanobis baseline is scored at four encoder layers with two poolings each and reported at its best, so it is given every advantage. The map-only column is the control: five statistics of the grid itself, no model involved. A detector that does not beat it has learned nothing about the planner. The last column bootstraps the gap between the combined detector and that control over evaluation instances: an interval straddling zero means the residual added nothing measurable.

| condition | instances | test pairs | residual AUC | residual accuracy | mahalanobis AUC | best layer/pooling | map-only AUC | map-only + residual | gap, 95 % CI |
|-----------|-----------|------------|--------------|-------------------|-----------------|--------------------|--------------|---------------------|--------------|
| `k1_opt` | 100 | 50 | 0.677 | 0.620 | 0.841 | `l2_mean` | 0.779 | 0.892 | +0.113 [+0.045, +0.178] |
| `k1_eff` | 100 | 50 | 0.778 | 0.650 | 0.833 | `l2_mean` | 0.634 | 0.871 | +0.238 [+0.142, +0.311] |
| `greedy_opt_b1` | 100 | 50 | 0.677 | 0.620 | 0.841 | `l2_mean` | 0.779 | 0.892 | +0.113 [+0.045, +0.178] |
| `greedy_opt_b4` | 100 | 50 | 0.798 | 0.700 | 0.978 | `l2_mean` | 0.910 | 0.920 | +0.010 [-0.030, +0.049] |
| `greedy_opt_b8` | 100 | 50 | 0.816 | 0.710 | 0.980 | `l2_mean` | 0.927 | 0.932 | +0.005 [-0.040, +0.052] |
| `greedy_eff_b1` | 94 | 47 | 0.778 | 0.649 | 0.792 | `l2_mean` | 0.611 | 0.861 | +0.251 [+0.165, +0.335] |
| `greedy_eff_b4` | 94 | 47 | 0.909 | 0.809 | 0.979 | `l2_mean` | 0.935 | 0.994 | +0.060 [+0.016, +0.115] |
| `greedy_eff_b8` | 94 | 47 | 0.956 | 0.883 | 0.986 | `l8_mean` | 0.929 | 0.996 | +0.067 [+0.017, +0.124] |
| `balanced_2` | 100 | 50 | 0.778 | 0.650 | 0.952 | `l2_mean` | 0.794 | 0.940 | +0.146 [+0.083, +0.206] |
| `blocked_2` | 100 | 50 | 0.734 | 0.690 | 0.955 | `l2_mean` | 0.918 | 0.979 | +0.061 [+0.011, +0.125] |

## Against the convolutional map-only control

A small convolutional network given the raw grid, trained on **random** single obstacle-additions to the 800 training maps — never on a selected damaging flip and never on a map it is scored on. This is the hardest model-free bar: it discovers whatever local pattern a flip leaves without being told to look for it, and a defender really could build it, since random flips cost nothing to generate.

| condition | convolutional control | + residual | gap, 95 % CI |
|-----------|-----------------------|------------|--------------|
| `k1_opt` | 0.919 | 0.975 | +0.056 [+0.012, +0.105] |
| `k1_eff` | 0.797 | 0.902 | +0.106 [+0.024, +0.182] |
| `greedy_opt_b1` | 0.919 | 0.975 | +0.056 [+0.012, +0.105] |
| `greedy_opt_b4` | 0.994 | 0.982 | -0.013 [-0.036, +0.002] |
| `greedy_opt_b8` | 0.990 | 0.977 | -0.012 [-0.044, +0.013] |
| `greedy_eff_b1` | 0.794 | 0.899 | +0.105 [+0.025, +0.181] |
| `greedy_eff_b4` | 0.996 | 0.993 | -0.003 [-0.012, +0.000] |
| `greedy_eff_b8` | 0.995 | 0.990 | -0.005 [-0.019, +0.000] |
| `balanced_2` | 0.966 | 0.981 | +0.016 [-0.018, +0.050] |
| `blocked_2` | 1.000 | 1.000 | +0.000 [-0.000, +0.000] |

## The harder version: no clean twin

Each evaluation map contributes either its clean map or its altered one, never both, so the detector cannot compare a map against itself. This is what a deployed monitor faces, and these are the realistic numbers.

| condition | residual AUC | map-only AUC | map-only + residual | gap, 95 % CI |
|-----------|--------------|--------------|---------------------|--------------|
| `k1_opt` | 0.662 | 0.724 | 0.923 | +0.199 [+0.066, +0.357] |
| `k1_eff` | 0.864 | 0.590 | 0.899 | +0.310 [+0.141, +0.488] |
| `greedy_opt_b1` | 0.662 | 0.724 | 0.923 | +0.199 [+0.066, +0.357] |
| `greedy_opt_b4` | 0.818 | 0.893 | 0.931 | +0.038 [-0.026, +0.114] |
| `greedy_opt_b8` | 0.843 | 0.896 | 0.925 | +0.029 [-0.039, +0.114] |
| `greedy_eff_b1` | 0.828 | 0.449 | 0.842 | +0.393 [+0.212, +0.577] |
| `greedy_eff_b4` | 0.957 | 0.955 | 1.000 | +0.045 [+0.000, +0.128] |
| `greedy_eff_b8` | 0.986 | 0.888 | 0.998 | +0.111 [+0.011, +0.236] |
| `balanced_2` | 0.875 | 0.783 | 0.962 | +0.178 [+0.061, +0.310] |
| `blocked_2` | 0.771 | 0.929 | 0.979 | +0.050 [-0.018, +0.143] |

