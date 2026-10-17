# Step 5 — surrogate fits of the cost map

Fitted on the train split (360,700 free reachable cells of 800 maps), scored on the test split (44,635 cells of 100 maps). R² is the share of the cost map's variance explained; the within-instance column removes each map's mean first, so a fit cannot score well merely by separating easy maps from hard ones.

| feature group | n | ridge R² | ridge within | GBM R² | GBM within |
|---------------|---|----------|--------------|--------|------------|
| `goal_to` | 3 | 0.007 | 0.004 | 0.074 | 0.077 |
| `heuristic` | 2 | 0.002 | 0.003 | — | — |
| `obstacle` | 7 | 0.022 | 0.020 | 0.153 | 0.160 |
| `geometric_h1` | 10 | 0.026 | 0.022 | 0.240 | 0.245 |
| `geometric` | 11 | 0.033 | 0.031 | 0.273 | 0.281 |
| `path` | 3 | 0.123 | 0.128 | 0.123 | 0.128 |
| `all` | 14 | 0.151 | 0.150 | 0.408 | 0.409 |

`goal_to` = the three distances to the goal; `heuristic` = the two features spanning the search's own heuristic; `obstacle` = distance transforms, densities and the two width proxies; **`geometric_h1` = `goal_to` + `obstacle`, which is hypothesis H1 exactly as stated**; `geometric` = the same plus `geodesic_from_start`; `path` = the three path indicators; `all` = everything.

**Why `geometric` is not a clean test of H1.** A cell lies on some optimal path exactly when `geodesic_from_start + geodesic_to_goal == optimum`, so a non-linear model handed both distances can rebuild the path indicator without being given it. The gap between `geometric_h1` and `geometric` is that leakage, and it is large — which is itself informative about what the encoder encodes.

## Residuals of the best fit (`all`, gradient boosting)

Grouped by distance to the nearest wall. PLAN.md asks for the free-neighbour topology partition, which is degenerate on this map family, so both are reported.

| wall zone | share of cells | mean \|residual\| | mean signed residual | mean cost |
|-----------|----------------|-------------------|----------------------|-----------|
| wall_adjacent | 31.0% | 0.1499 | -0.0012 | 0.564 |
| near_wall | 26.1% | 0.1794 | -0.0052 | 0.511 |
| interior | 42.9% | 0.1848 | -0.0055 | 0.575 |

| topology class | share of cells | mean \|residual\| | mean signed residual | mean cost |
|----------------|----------------|-------------------|----------------------|-----------|
| corridor | 1.8% | 0.0909 | -0.0048 | 0.628 |
| junction | 28.0% | 0.1537 | -0.0015 | 0.558 |
| open | 70.2% | 0.1821 | -0.0051 | 0.552 |

Residual maps: `e1_surrogate_residuals.png`.

## Sufficiency test — can a surrogate cost map plan?

R² measures reconstruction, not behaviour: a cost-map difference only matters if it changes where the search goes. Here the surrogate's prediction is assembled back into a cost map and handed to the *same* search module the encoder feeds. `classical_unit_cost` is the same module with a flat cost map, i.e. ordinary A\*.

| cost map | Opt (exactly optimal) | mean Opt ratio | mean Exp ratio | Hmean |
|----------|----------------------|----------------|----------------|-------|
| `neural_encoder` | 0.830 | 1.0154 | 0.578 | 0.5728 |
| `surrogate_geometric_h1` | 0.620 | 1.0366 | 0.755 | 0.4129 |
| `surrogate_all` | 0.920 | 1.0074 | 0.627 | 0.5521 |
| `classical_unit_cost` | 1.000 | 1.0000 | 1.000 | 0.0000 |

