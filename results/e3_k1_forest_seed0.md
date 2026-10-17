# Step 7 — mediation of single-flip damage (E3)

Split `test`, seed 0. The alterations are the worst single-cell flips found by the exhaustive sweep of step 6 — one per map per objective — evaluated under all four cost/obstacle conditions (195 swaps, 8 s).

Each condition is scored against the optimum and the classical-A\* effort of **its own obstacle mask**, since the mask defines the problem actually being solved.

## Optimality objective (metric `OPT`)

100 alterations; 92 of them actually changed `OPT` and so can be decomposed.

| route pattern | alterations | share |
|---------------|-------------|-------|
| encoder | 28 | 28% |
| search | 35 | 35% |
| both | 1 | 1% |
| interaction | 28 | 28% |
| no damage | 8 | 8% |

Mediation fractions over the 92 damaged alterations. The distribution is bimodal — an individual failure is usually reproduced almost entirely by one route or by neither — so the share at each end says more than the average:

| route | mean | median | IQR | explains it alone (≥ 0.75) | contributes nothing (≤ 0.25) |
|-------|------|--------|-----|----------------------------|------------------------------|
| encoder | +0.349 | +0.00 | +0.00 – +1.00 | 32% | 62% |
| search | +0.397 | +0.00 | +0.00 – +1.00 | 39% | 51% |

The two routes sum to 0.746 on average (median 1.000). A sum near 1 means the decomposition is nearly additive — the two routes partition the damage; a sum well below 1 means the rest is interaction, needing both changes at once.

### Split by failure mode

Two different things are being called damage here. When the flip leaves the map's optimum unchanged, the planner took a worse path on a problem whose answer did not move. When the flip opens a new shortcut the optimum drops, and the planner is penalised for not using a route that exists only under the altered mask — the mask being exactly what the search route varies, so a search-dominated reading there is close to definitional. Pooling the two is misleading:

| failure mode | n | encoder mean | encoder dominant | search mean | search dominant |
|--------------|---|--------------|------------------|-------------|-----------------|
| optimum unchanged | 79 | +0.407 | 37% | +0.310 | 30% |
| optimum changed | 13 | +0.000 | 0% | +0.923 | 92% |

Robustness check on raw counts (path length in moves, or expanded cells) rather than the ratios, which have a per-condition denominator: encoder mean +0.378, search mean +0.349, over 85 alterations. The two need not agree exactly — when a flip changes the optimum, a ratio can worsen while the raw path length improves.

## Efficiency objective (metric `EXP`)

95 alterations; 74 of them actually changed `EXP` and so can be decomposed.

| route pattern | alterations | share |
|---------------|-------------|-------|
| encoder | 23 | 24% |
| search | 36 | 38% |
| both | 1 | 1% |
| interaction | 14 | 15% |
| no damage | 21 | 22% |

Mediation fractions over the 74 damaged alterations. The distribution is bimodal — an individual failure is usually reproduced almost entirely by one route or by neither — so the share at each end says more than the average:

| route | mean | median | IQR | explains it alone (≥ 0.75) | contributes nothing (≤ 0.25) |
|-------|------|--------|-----|----------------------------|------------------------------|
| encoder | +0.418 | +0.31 | +0.00 – +0.97 | 32% | 49% |
| search | +0.622 | +0.77 | +0.00 – +1.00 | 50% | 43% |

The two routes sum to 1.040 on average (median 1.000). A sum near 1 means the decomposition is nearly additive — the two routes partition the damage; a sum well below 1 means the rest is interaction, needing both changes at once.

### Split by failure mode

Two different things are being called damage here. When the flip leaves the map's optimum unchanged, the planner took a worse path on a problem whose answer did not move. When the flip opens a new shortcut the optimum drops, and the planner is penalised for not using a route that exists only under the altered mask — the mask being exactly what the search route varies, so a search-dominated reading there is close to definitional. Pooling the two is misleading:

| failure mode | n | encoder mean | encoder dominant | search mean | search dominant |
|--------------|---|--------------|------------------|-------------|-----------------|
| optimum unchanged | 56 | +0.521 | 43% | +0.487 | 38% |
| optimum changed | 18 | +0.096 | 0% | +1.039 | 89% |

Robustness check on raw counts (path length in moves, or expanded cells) rather than the ratios, which have a per-condition denominator: encoder mean +0.470, search mean +0.415, over 62 alterations. The two need not agree exactly — when a flip changes the optimum, a ratio can worsen while the raw path length improves.

Plot: `e3_k1_mediation_forest_seed0.png`. Each point is one alteration; the axes are the two mediation fractions. A point near (1, 0) is encoder-mediated, near (0, 1) is search-mediated, near the origin means neither route reproduces the damage on its own.

## Caveats

* The off-diagonal conditions are off-distribution by construction: the encoder never produced that cost map for that obstacle mask. A strange `C1O0` can be the mismatch rather than the encoder's distortion.
* The route labels use a tolerance of 0.25 around 0 and 1; they summarise the scatter, they are not a test.
* These are the *worst* flips per map, selected by the same metric they are then decomposed on, so they are not a random sample of alterations.

