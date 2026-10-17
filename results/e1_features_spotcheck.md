# Step 4 — feature library spot-check

Split `test`, seed 0, 100 instances, 14 features. Figure: `e1_features_spotcheck.png` (instances 0, 1, 2).

Definitions and conventions are in `docs/decisions.md` (2026-10-17, "Feature library definitions").

## Correlation with the cost map (diagnostic, not the step-5 fit)

| feature | pooled r | mean within-instance r | sd across instances |
|---------|----------|------------------------|---------------------|
| `on_any_optimal_path` | -0.327 | -0.339 | 0.107 |
| `on_reference_path` | -0.248 | -0.248 | 0.068 |
| `on_solver_path` | -0.205 | -0.205 | 0.080 |
| `corridor_span_min` | 0.086 | 0.062 | 0.096 |
| `geodesic_from_start` | 0.063 | 0.062 | 0.127 |
| `corridor_width_min4` | 0.050 | 0.045 | 0.065 |
| `dist_to_obstacle_euclidean` | 0.047 | 0.041 | 0.064 |
| `dist_to_obstacle_chebyshev` | 0.045 | 0.039 | 0.064 |
| `obstacle_density_3x3` | 0.034 | 0.038 | 0.047 |
| `free_neighbours` | -0.034 | -0.038 | 0.047 |
| `geodesic_to_goal` | 0.033 | 0.022 | 0.123 |
| `chebyshev_to_goal` | 0.018 | 0.020 | 0.128 |
| `euclidean_to_goal` | 0.007 | 0.010 | 0.129 |
| `obstacle_density_5x5` | -0.002 | 0.003 | 0.053 |

The cost map's spread is almost entirely *within* instances (sd 0.288) rather than between them (sd 0.033), so the encoder output is spatially structured rather than a per-instance level shift. Values span 0.000–1.000 (mean 0.555), i.e. the sigmoid is not saturated.

**Early signal, stated carefully.** The three path indicators are the strongest linear correlates of the cost map, while every distance-to-goal feature is nearly uncorrelated with it (|r| ≤ 0.03 pooled *and* within instance). That is the direction of PLAN.md's *competing* hypothesis — a path-prediction map — rather than H1's smoothed geometric heuristic. Two reasons this is not yet the answer: these are single-feature Pearson correlations, so they say nothing about what the features explain *jointly* or non-linearly; and the sign matters — cells on optimal paths carry *lower* cost, which is what steering the search toward them requires. Step 5's three fits (geometric-only / path-only / both, fit on train, R² on test) are the actual test.

