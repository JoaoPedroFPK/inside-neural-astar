# Step 10 — dispersion across training seeds

3 model(s) compared on the same instances and the same fixed starts. A difference between conditions is only a result if it is larger than the difference between seeds.

| model | epoch | exactly optimal | mean Exp | Hmean | R² geometry | R² path | R² all | sufficiency Opt |
|-------|-------|-----------------|----------|-------|-------------|---------|--------|-----------------|
| `forest_seed0` | 19 | 0.540 | 0.816 | 0.412 | 0.205 | 0.035 | 0.315 | 0.810 |
| `forest_seed1` | 23 | 0.560 | 0.750 | 0.446 | 0.301 | 0.064 | 0.412 | 0.770 |
| `forest_seed2` | 27 | 0.530 | 0.760 | 0.434 | 0.238 | 0.043 | 0.359 | 0.720 |

## Is the planner equally fragile, model to model?

From each model's own exhaustive sweep of all ~102 000 single-cell alterations.

| model | maps with a damaging cell | median damaging cells | worst Opt |
|-------|---------------------------|-----------------------|-----------|
| `forest_seed0` | 85.2% | 0.73% | 1.474 |
| `forest_seed1` | 87.5% | 0.98% | 1.478 |
| `forest_seed2` | 84.9% | 0.88% | 1.393 |


## The same alterations, decomposed by every model

Repeating step 7 per model uses each model's own worst flips, which mixes two sources of variance: the models differ and the alterations differ. Here every model is run on one fixed set — the flips chosen as worst for the shipped checkpoint — which isolates the model difference.

| model | n damaged | path: encoder | path: search | n damaged | effort: encoder | effort: search |
|-------|-----------|---------------|--------------|-----------|-----------------|----------------|
| `forest_seed0` | 79 | +0.407 | +0.310 | 56 | +0.521 | +0.487 |
| `forest_seed1` | 34 | +0.358 | +0.583 | 48 | +0.322 | +0.758 |
| `forest_seed2` | 36 | +0.319 | +0.634 | 49 | +0.206 | +0.805 |

**Read the `n` columns before the fractions.** The alterations are the same, but how many of them actually damage each model is not: a flip chosen as worst for one model often leaves another untouched. So the fractions are still averages over different subsets, and only the comparison of *which route dominates* is meaningful — not the exact values.


## Mediation per model (clean failure mode only)

Each model has its own worst flips, so each needs its own sweep and swap. Restricted to alterations that did not change the map's optimum.

| model | path: encoder | path: search | effort: encoder | effort: search |
|-------|---------------|--------------|-----------------|----------------|
| `forest_seed0` | +0.407 | +0.310 | +0.521 | +0.487 |
| `forest_seed1` | +0.544 | +0.346 | +0.519 | +0.451 |
| `forest_seed2` | +0.493 | +0.382 | +0.355 | +0.695 |

## Do the models agree on the cost map itself?

The cost map is the object this project interprets. Correlations over free, reachable cells; the within-map column removes any contribution from models simply disagreeing about which maps are hard.

| pair | pooled r | within-map r | mean \|difference\| |
|------|----------|--------------|---------------------|
| forest_seed0 vs forest_seed1 | 0.712 | 0.711 | 0.147 |
| forest_seed0 vs forest_seed2 | 0.672 | 0.670 | 0.152 |
| forest_seed1 vs forest_seed2 | 0.677 | 0.674 | 0.154 |

## Do the models break on the same cells?

Every model's sweep labels all ~102 000 single-cell alterations as damaging or not. Comparing the labels separates a property of the map — this cell is dangerous for any planner — from a property of the model.

| pair | damaging for each | damaging for both | share | vs chance | same worst flip |
|------|-------------------|-------------------|-------|-----------|-----------------|
| forest_seed0 vs forest_seed1 | 1,895 / 2,653 | 707 | 37% | **14x** | 23/100 maps |
| forest_seed0 vs forest_seed2 | 1,895 / 2,449 | 499 | 26% | **11x** | 25/100 maps |
| forest_seed1 vs forest_seed2 | 2,653 / 2,449 | 807 | 30% | **13x** | 21/100 maps |


## Dispersion

| quantity | mean | std | min | max |
|---|---|---|---|---|
| `checkpoint_epoch` | 23.0000 | 4.0000 | 19.0000 | 27.0000 |
| `planning.p_opt` | 0.5433 | 0.0153 | 0.5300 | 0.5600 |
| `planning.mean_opt` | 1.0485 | 0.0034 | 1.0455 | 1.0522 |
| `planning.mean_exp` | 0.7756 | 0.0353 | 0.7504 | 0.8160 |
| `planning.h_mean` | 0.4306 | 0.0174 | 0.4118 | 0.4460 |
| `surrogate_r2.geometric_h1_gbm` | 0.2478 | 0.0488 | 0.2046 | 0.3007 |
| `surrogate_r2.path_gbm` | 0.0475 | 0.0149 | 0.0353 | 0.0641 |
| `surrogate_r2.all_gbm` | 0.3622 | 0.0488 | 0.3149 | 0.4123 |
| `sufficiency_all.p_opt` | 0.7667 | 0.0451 | 0.7200 | 0.8100 |
| `sufficiency_all.mean_opt` | 1.0187 | 0.0049 | 1.0141 | 1.0239 |
| `sufficiency_all.mean_exp` | 0.7217 | 0.0111 | 0.7101 | 0.7322 |
| `sufficiency_all.h_mean` | 0.4896 | 0.0077 | 0.4849 | 0.4985 |
| `mediation_optimality.n` | 78.3333 | 4.0415 | 74.0000 | 82.0000 |
| `mediation_optimality.encoder` | 0.4814 | 0.0695 | 0.4068 | 0.5443 |
| `mediation_optimality.search` | 0.3460 | 0.0358 | 0.3101 | 0.3818 |
| `mediation_efficiency.n` | 57.6667 | 3.7859 | 55.0000 | 62.0000 |
| `mediation_efficiency.encoder` | 0.4652 | 0.0954 | 0.3551 | 0.5212 |
| `mediation_efficiency.search` | 0.5448 | 0.1318 | 0.4513 | 0.6955 |
| `fragility.maps_with_a_damaging_cell` | 0.8586 | 0.0142 | 0.8491 | 0.8750 |
| `fragility.median_damaging_cells` | 0.0086 | 0.0012 | 0.0073 | 0.0098 |
| `fragility.mean_damaging_cells` | 0.0228 | 0.0038 | 0.0185 | 0.0260 |
| `fragility.worst_opt` | 1.4483 | 0.0480 | 1.3929 | 1.4783 |
| `fragility.median_worst_opt` | 1.0562 | 0.0106 | 1.0455 | 1.0667 |
| `shared_optimality.n` | 49.6667 | 25.4231 | 34.0000 | 79.0000 |
| `shared_optimality.encoder` | 0.3613 | 0.0438 | 0.3194 | 0.4068 |
| `shared_optimality.search` | 0.5092 | 0.1743 | 0.3101 | 0.6343 |
| `shared_efficiency.n` | 51.0000 | 4.3589 | 48.0000 | 56.0000 |
| `shared_efficiency.encoder` | 0.3496 | 0.1596 | 0.2055 | 0.5212 |
| `shared_efficiency.search` | 0.6835 | 0.1714 | 0.4875 | 0.8054 |

