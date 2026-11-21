# Step 13 — gradient-guided flip attack

Split `test`, seed 0, 100 maps, pool 64 cells per round, budgets up to 8. Run time 101.5 min.

## 1. Does the gradient rank single flips?

Against the exhaustive sweep, which knows the true effect of every single flip. `top 16` is the 16 cells with the largest first-order score.

| objective | maps | Spearman (abs score) | damaging share, top 16 | damaging share, all cells | best of top 16 / worst cell | best of 16 random / worst cell | top 16 holds the worst cell |
|---|---|---|---|---|---|---|---|
| efficiency (all) | 100 | 0.201 | 0.623 | 0.488 | 0.717 | 0.634 | 0.09 |
| optimality (all) | 100 | 0.109 | 0.220 | 0.181 | 0.948 | 0.933 | 0.36 |
| efficiency (held out) | 90 | 0.204 | 0.617 | 0.478 | 0.722 | 0.645 | 0.09 |
| optimality (held out) | 90 | 0.109 | 0.216 | 0.178 | 0.949 | 0.934 | 0.37 |

## 2. Damage at each budget

`gradient`: pool re-ranked each round by the gradient. `random`: the same search with a random pool of the same size. `k1_sweep`: step 9's greedy, pool read off the exhaustive sweep (an attacker who has already tried every cell).

| objective | budget | source | maps | mean Opt | baseline-optimal maps broken | mean Exp | mean Exp / baseline |
|---|---|---|---|---|---|---|---|
| efficiency | 1 | `gradient` | 100 | – | – | 0.866 | 1.520 |
| efficiency | 2 | `gradient` | 100 | – | – | 1.085 | 1.960 |
| efficiency | 4 | `gradient` | 100 | – | – | 1.516 | 2.742 |
| efficiency | 8 | `gradient` | 100 | – | – | 2.061 | 3.768 |
| efficiency | 1 | `k1_sweep` | 94 | – | – | 1.152 | 2.020 |
| efficiency | 2 | `k1_sweep` | 94 | – | – | 1.452 | 2.615 |
| efficiency | 4 | `k1_sweep` | 94 | – | – | 1.756 | 3.330 |
| efficiency | 8 | `k1_sweep` | 94 | – | – | 1.981 | 3.887 |
| efficiency | 1 | `random` | 95 | – | – | 0.698 | 1.292 |
| efficiency | 2 | `random` | 95 | – | – | 0.848 | 1.533 |
| efficiency | 4 | `random` | 95 | – | – | 1.280 | 2.254 |
| efficiency | 8 | `random` | 95 | – | – | 1.673 | 2.973 |
| optimality | 1 | `gradient` | 100 | 1.079 | 0.627 (83) | – | – |
| optimality | 2 | `gradient` | 100 | 1.112 | 0.747 (83) | – | – |
| optimality | 4 | `gradient` | 100 | 1.143 | 0.807 (83) | – | – |
| optimality | 8 | `gradient` | 100 | 1.199 | 0.940 (83) | – | – |
| optimality | 1 | `k1_sweep` | 100 | 1.110 | 0.855 (83) | – | – |
| optimality | 2 | `k1_sweep` | 100 | 1.137 | 0.855 (83) | – | – |
| optimality | 4 | `k1_sweep` | 100 | 1.156 | 0.855 (83) | – | – |
| optimality | 8 | `k1_sweep` | 100 | 1.167 | 0.855 (83) | – | – |
| optimality | 1 | `random` | 100 | 1.054 | 0.434 (83) | – | – |
| optimality | 2 | `random` | 100 | 1.079 | 0.602 (83) | – | – |
| optimality | 4 | `random` | 100 | 1.105 | 0.675 (83) | – | – |
| optimality | 8 | `random` | 100 | 1.145 | 0.723 (83) | – | – |

## 3. Paired gaps (per-map mean difference in Opt or Exp, 95 % bootstrap CI)

| objective | budget | comparison | maps | gap | 95 % CI | held-out gap | held-out CI |
|---|---|---|---|---|---|---|---|
| optimality | 1 | gradient − random | 100 | +0.025 | [+0.015, +0.037] | +0.023 | [+0.013, +0.036] |
| optimality | 1 | gradient − k1_sweep | 100 | -0.031 | [-0.041, -0.022] | -0.031 | [-0.042, -0.022] |
| optimality | 2 | gradient − random | 100 | +0.033 | [+0.021, +0.046] | +0.033 | [+0.020, +0.049] |
| optimality | 2 | gradient − k1_sweep | 100 | -0.025 | [-0.039, -0.011] | -0.025 | [-0.040, -0.008] |
| optimality | 4 | gradient − random | 100 | +0.038 | [+0.024, +0.054] | +0.037 | [+0.021, +0.056] |
| optimality | 4 | gradient − k1_sweep | 100 | -0.013 | [-0.030, +0.004] | -0.014 | [-0.033, +0.005] |
| optimality | 8 | gradient − random | 100 | +0.054 | [+0.033, +0.076] | +0.052 | [+0.031, +0.075] |
| optimality | 8 | gradient − k1_sweep | 100 | +0.033 | [+0.014, +0.052] | +0.033 | [+0.013, +0.055] |
| efficiency | 1 | gradient − random | 95 | +0.181 | [+0.070, +0.308] | +0.159 | [+0.041, +0.300] |
| efficiency | 1 | gradient − k1_sweep | 94 | -0.270 | [-0.405, -0.156] | -0.285 | [-0.441, -0.160] |
| efficiency | 2 | gradient − random | 95 | +0.236 | [+0.084, +0.408] | +0.254 | [+0.094, +0.439] |
| efficiency | 2 | gradient − k1_sweep | 94 | -0.365 | [-0.557, -0.195] | -0.348 | [-0.565, -0.162] |
| efficiency | 4 | gradient − random | 95 | +0.236 | [-0.002, +0.485] | +0.307 | [+0.117, +0.552] |
| efficiency | 4 | gradient − k1_sweep | 94 | -0.247 | [-0.460, -0.052] | -0.204 | [-0.427, -0.009] |
| efficiency | 8 | gradient − random | 95 | +0.397 | [+0.099, +0.692] | +0.451 | [+0.196, +0.731] |
| efficiency | 8 | gradient − k1_sweep | 94 | +0.080 | [-0.216, +0.363] | +0.153 | [-0.155, +0.437] |

Planner evaluations per map and objective: gradient and random 512 (plus 8 gradient passes for `gradient`); `k1_sweep` the same plus the 1,021 of the exhaustive sweep.
