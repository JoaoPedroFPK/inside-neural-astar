# Step 7 — mediation of single-flip damage (E3)

Split `test`, seed 0. The alterations are the worst single-cell flips found by the exhaustive sweep of step 6 — one per map per objective — evaluated under all four cost/obstacle conditions (200 swaps, 11 s).

Each condition is scored against the optimum and the classical-A\* effort of **its own obstacle mask**, since the mask defines the problem actually being solved.

## Optimality objective (metric `OPT`)

100 alterations; 89 of them actually changed `OPT` and so can be decomposed.

| route pattern | alterations | share |
|---------------|-------------|-------|
| encoder | 30 | 30% |
| search | 39 | 39% |
| both | 1 | 1% |
| interaction | 19 | 19% |
| no damage | 11 | 11% |

Mediation fractions over the 89 damaged alterations. The distribution is bimodal — an individual failure is usually reproduced almost entirely by one route or by neither — so the share at each end says more than the average:

| route | mean | median | IQR | explains it alone (≥ 0.75) | contributes nothing (≤ 0.25) |
|-------|------|--------|-----|----------------------------|------------------------------|
| encoder | +0.384 | +0.00 | +0.00 – +1.00 | 35% | 58% |
| search | +0.501 | +0.50 | +0.00 – +1.00 | 45% | 45% |

The two routes sum to 0.885 on average (median 1.000). A sum near 1 means the decomposition is nearly additive — the two routes partition the damage; a sum well below 1 means the rest is interaction, needing both changes at once.

### Split by failure mode

Two different things are being called damage here. When the flip leaves the map's optimum unchanged, the planner took a worse path on a problem whose answer did not move. When the flip opens a new shortcut the optimum drops, and the planner is penalised for not using a route that exists only under the altered mask — the mask being exactly what the search route varies, so a search-dominated reading there is close to definitional. Pooling the two is misleading:

| failure mode | n | encoder mean | encoder dominant | search mean | search dominant |
|--------------|---|--------------|------------------|-------------|-----------------|
| optimum unchanged | 70 | +0.463 | 43% | +0.437 | 40% |
| optimum changed | 19 | +0.090 | 5% | +0.739 | 63% |

Robustness check on raw counts (path length in moves, or expanded cells) rather than the ratios, which have a per-condition denominator: encoder mean +0.478, search mean +0.425, over 72 alterations. The two need not agree exactly — when a flip changes the optimum, a ratio can worsen while the raw path length improves.

## Efficiency objective (metric `EXP`)

100 alterations; 88 of them actually changed `EXP` and so can be decomposed.

| route pattern | alterations | share |
|---------------|-------------|-------|
| encoder | 12 | 12% |
| search | 35 | 35% |
| both | 1 | 1% |
| interaction | 40 | 40% |
| no damage | 12 | 12% |

Mediation fractions over the 88 damaged alterations. The distribution is bimodal — an individual failure is usually reproduced almost entirely by one route or by neither — so the share at each end says more than the average:

| route | mean | median | IQR | explains it alone (≥ 0.75) | contributes nothing (≤ 0.25) |
|-------|------|--------|-----|----------------------------|------------------------------|
| encoder | +0.207 | +0.02 | +0.00 – +0.26 | 15% | 75% |
| search | +0.567 | +0.56 | +0.10 – +1.00 | 41% | 32% |

The two routes sum to 0.774 on average (median 0.978). A sum near 1 means the decomposition is nearly additive — the two routes partition the damage; a sum well below 1 means the rest is interaction, needing both changes at once.

### Split by failure mode

Two different things are being called damage here. When the flip leaves the map's optimum unchanged, the planner took a worse path on a problem whose answer did not move. When the flip opens a new shortcut the optimum drops, and the planner is penalised for not using a route that exists only under the altered mask — the mask being exactly what the search route varies, so a search-dominated reading there is close to definitional. Pooling the two is misleading:

| failure mode | n | encoder mean | encoder dominant | search mean | search dominant |
|--------------|---|--------------|------------------|-------------|-----------------|
| optimum unchanged | 49 | +0.150 | 10% | +0.327 | 22% |
| optimum changed | 39 | +0.279 | 21% | +0.868 | 64% |

Robustness check on raw counts (path length in moves, or expanded cells) rather than the ratios, which have a per-condition denominator: encoder mean +0.595, search mean +0.242, over 68 alterations. The two need not agree exactly — when a flip changes the optimum, a ratio can worsen while the raw path length improves.

Plot: `e3_k1_mediation_mazes_seed0.png`. Each point is one alteration; the axes are the two mediation fractions. A point near (1, 0) is encoder-mediated, near (0, 1) is search-mediated, near the origin means neither route reproduces the damage on its own.

## Caveats

* The off-diagonal conditions are off-distribution by construction: the encoder never produced that cost map for that obstacle mask. A strange `C1O0` can be the mismatch rather than the encoder's distortion.
* The route labels use a tolerance of 0.25 around 0 and 1; they summarise the scatter, they are not a test.
* These are the *worst* flips per map, selected by the same metric they are then decomposed on, so they are not a random sample of alterations.

