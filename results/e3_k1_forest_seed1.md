# Step 7 — mediation of single-flip damage (E3)

Split `test`, seed 0. The alterations are the worst single-cell flips found by the exhaustive sweep of step 6 — one per map per objective — evaluated under all four cost/obstacle conditions (193 swaps, 8 s).

Each condition is scored against the optimum and the classical-A\* effort of **its own obstacle mask**, since the mask defines the problem actually being solved.

## Optimality objective (metric `OPT`)

100 alterations; 93 of them actually changed `OPT` and so can be decomposed.

| route pattern | alterations | share |
|---------------|-------------|-------|
| encoder | 41 | 41% |
| search | 33 | 33% |
| both | 1 | 1% |
| interaction | 18 | 18% |
| no damage | 7 | 7% |

Mediation fractions over the 93 damaged alterations. The distribution is bimodal — an individual failure is usually reproduced almost entirely by one route or by neither — so the share at each end says more than the average:

| route | mean | median | IQR | explains it alone (≥ 0.75) | contributes nothing (≤ 0.25) |
|-------|------|--------|-----|----------------------------|------------------------------|
| encoder | +0.500 | +0.50 | +0.00 – +1.00 | 45% | 44% |
| search | +0.433 | +0.25 | +0.00 – +1.00 | 37% | 51% |

The two routes sum to 0.933 on average (median 1.000). A sum near 1 means the decomposition is nearly additive — the two routes partition the damage; a sum well below 1 means the rest is interaction, needing both changes at once.

### Split by failure mode

Two different things are being called damage here. When the flip leaves the map's optimum unchanged, the planner took a worse path on a problem whose answer did not move. When the flip opens a new shortcut the optimum drops, and the planner is penalised for not using a route that exists only under the altered mask — the mask being exactly what the search route varies, so a search-dominated reading there is close to definitional. Pooling the two is misleading:

| failure mode | n | encoder mean | encoder dominant | search mean | search dominant |
|--------------|---|--------------|------------------|-------------|-----------------|
| optimum unchanged | 82 | +0.544 | 50% | +0.346 | 30% |
| optimum changed | 11 | +0.171 | 9% | +1.078 | 82% |

Robustness check on raw counts (path length in moves, or expanded cells) rather than the ratios, which have a per-condition denominator: encoder mean +0.542, search mean +0.353, over 86 alterations. The two need not agree exactly — when a flip changes the optimum, a ratio can worsen while the raw path length improves.

## Efficiency objective (metric `EXP`)

93 alterations; 71 of them actually changed `EXP` and so can be decomposed.

| route pattern | alterations | share |
|---------------|-------------|-------|
| encoder | 28 | 30% |
| search | 34 | 37% |
| both | 1 | 1% |
| interaction | 8 | 9% |
| no damage | 22 | 24% |

Mediation fractions over the 71 damaged alterations. The distribution is bimodal — an individual failure is usually reproduced almost entirely by one route or by neither — so the share at each end says more than the average:

| route | mean | median | IQR | explains it alone (≥ 0.75) | contributes nothing (≤ 0.25) |
|-------|------|--------|-----|----------------------------|------------------------------|
| encoder | +0.426 | +0.28 | +0.00 – +0.96 | 41% | 48% |
| search | +0.586 | +0.73 | +0.01 – +1.00 | 49% | 42% |

The two routes sum to 1.012 on average (median 1.000). A sum near 1 means the decomposition is nearly additive — the two routes partition the damage; a sum well below 1 means the rest is interaction, needing both changes at once.

### Split by failure mode

Two different things are being called damage here. When the flip leaves the map's optimum unchanged, the planner took a worse path on a problem whose answer did not move. When the flip opens a new shortcut the optimum drops, and the planner is penalised for not using a route that exists only under the altered mask — the mask being exactly what the search route varies, so a search-dominated reading there is close to definitional. Pooling the two is misleading:

| failure mode | n | encoder mean | encoder dominant | search mean | search dominant |
|--------------|---|--------------|------------------|-------------|-----------------|
| optimum unchanged | 55 | +0.519 | 51% | +0.451 | 36% |
| optimum changed | 16 | +0.105 | 6% | +1.049 | 94% |

Robustness check on raw counts (path length in moves, or expanded cells) rather than the ratios, which have a per-condition denominator: encoder mean +0.576, search mean +0.398, over 55 alterations. The two need not agree exactly — when a flip changes the optimum, a ratio can worsen while the raw path length improves.

Plot: `e3_k1_mediation_forest_seed1.png`. Each point is one alteration; the axes are the two mediation fractions. A point near (1, 0) is encoder-mediated, near (0, 1) is search-mediated, near the origin means neither route reproduces the damage on its own.

## Caveats

* The off-diagonal conditions are off-distribution by construction: the encoder never produced that cost map for that obstacle mask. A strange `C1O0` can be the mismatch rather than the encoder's distortion.
* The route labels use a tolerance of 0.25 around 0 and 1; they summarise the scatter, they are not a test.
* These are the *worst* flips per map, selected by the same metric they are then decomposed on, so they are not a random sample of alterations.

