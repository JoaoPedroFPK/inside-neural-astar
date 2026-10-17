# Step 5 — surrogate fits of the cost map

Fitted on the train split (677,704 free reachable cells of 800 maps), scored on the test split (84,873 cells of 100 maps). R² is the share of the cost map's variance explained; the within-instance column removes each map's mean first, so a fit cannot score well merely by separating easy maps from hard ones.

| feature group | n | ridge R² | ridge within | GBM R² | GBM within |
|---------------|---|----------|--------------|--------|------------|
| `goal_to` | 3 | 0.002 | 0.001 | 0.074 | 0.075 |
| `heuristic` | 2 | 0.001 | 0.001 | — | — |
| `obstacle` | 7 | 0.025 | 0.025 | 0.123 | 0.123 |
| `geometric_h1` | 10 | 0.026 | 0.025 | 0.205 | 0.205 |
| `geometric` | 11 | 0.036 | 0.034 | 0.278 | 0.278 |
| `path` | 3 | 0.034 | 0.035 | 0.035 | 0.036 |
| `all` | 14 | 0.062 | 0.061 | 0.315 | 0.315 |

`goal_to` = the three distances to the goal; `heuristic` = the two features spanning the search's own heuristic; `obstacle` = distance transforms, densities and the two width proxies; **`geometric_h1` = `goal_to` + `obstacle`, which is hypothesis H1 exactly as stated**; `geometric` = the same plus `geodesic_from_start`; `path` = the three path indicators; `all` = everything.

**Why `geometric` is not a clean test of H1.** A cell lies on some optimal path exactly when `geodesic_from_start + geodesic_to_goal == optimum`, so a non-linear model handed both distances can rebuild the path indicator without being given it. The gap between `geometric_h1` and `geometric` is that leakage, and it is large — which is itself informative about what the encoder encodes.

## Residuals of the best fit (`all`, gradient boosting)

Grouped by distance to the nearest wall. PLAN.md asks for the free-neighbour topology partition, which is degenerate on this map family, so both are reported.

| wall zone | share of cells | mean \|residual\| | mean signed residual | mean cost |
|-----------|----------------|-------------------|----------------------|-----------|
| wall_adjacent | 33.8% | 0.1345 | -0.0035 | 0.532 |
| near_wall | 29.1% | 0.1637 | -0.0008 | 0.547 |
| interior | 37.1% | 0.1434 | -0.0003 | 0.553 |

| topology class | share of cells | mean \|residual\| | mean signed residual | mean cost |
|----------------|----------------|-------------------|----------------------|-----------|
| dead_end | 0.0% | 0.1180 | +0.0207 | 0.703 |
| corridor | 2.1% | 0.0887 | -0.0034 | 0.617 |
| junction | 28.1% | 0.1315 | -0.0036 | 0.521 |
| open | 69.8% | 0.1539 | -0.0006 | 0.551 |

Residual maps: `e1_surrogate_residuals_forest_seed0.png`.

## Sufficiency test — can a surrogate cost map plan?

R² measures reconstruction, not behaviour: a cost-map difference only matters if it changes where the search goes. Here the surrogate's prediction is assembled back into a cost map and handed to the *same* search module the encoder feeds. `classical_unit_cost` is the same module with a flat cost map, i.e. ordinary A\*.

| cost map | Opt (exactly optimal) | mean Opt ratio | mean Exp ratio | Hmean |
|----------|----------------------|----------------|----------------|-------|
| `neural_encoder` | 0.540 | 1.0522 | 0.816 | 0.4118 |
| `surrogate_geometric_h1` | 0.550 | 1.0384 | 0.829 | 0.3873 |
| `surrogate_all` | 0.810 | 1.0141 | 0.723 | 0.4849 |
| `classical_unit_cost` | 1.000 | 1.0000 | 1.000 | 0.0000 |

### Control: is the surrogate just weaker guidance?

Weaker guidance makes the search behave more like plain A\*: more optimal, less efficient. If that were the whole story, the surrogate would sit on the curve traced by scaling the encoder's own output toward a constant (`0.5 + a * (cost - 0.5)`), which keeps its shape and reduces only its strength.

| scale `a` | exactly optimal | mean Exp ratio | Hmean |
|-----------|-----------------|----------------|-------|
| 1.00 | 0.540 | 0.816 | 0.4118 |
| 0.75 | 0.550 | 0.826 | 0.4123 |
| 0.50 | 0.550 | 0.840 | 0.4084 |
| 0.25 | 0.550 | 0.863 | 0.4022 |
| 0.10 | 0.570 | 0.868 | 0.4018 |
| 0.00 | 0.700 | 0.694 | 0.4257 |

The surrogate reaches 0.810 optimality at 0.723 of classical effort. It is **strictly better on both axes** than 5 point(s) on the weakening curve, so its advantage is not merely that it guides less — the structure it imposes is better for path quality at a comparable search cost.

Note also that scaling to `a = 0` is **not** classical A\*: a constant cost still enters `g` and `h`, changing their balance and making the search greedier. That is why the bottom row is less optimal than the flat-cost-map row above.

