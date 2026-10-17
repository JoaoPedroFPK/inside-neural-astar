# Step 10 — dispersion across training seeds

4 model(s) compared on the same instances and the same fixed starts. A difference between conditions is only a result if it is larger than the difference between seeds.

| model | epoch | exactly optimal | mean Exp | Hmean | R² geometry | R² path | R² all | sufficiency Opt |
|-------|-------|-----------------|----------|-------|-------------|---------|--------|-----------------|
| `pretrained` | 33 | 0.830 | 0.578 | 0.573 | 0.240 | 0.123 | 0.408 | 0.920 |
| `mazes_seed0` | 23 | 0.800 | 0.605 | 0.553 | 0.274 | 0.119 | 0.437 | 0.930 |
| `mazes_seed1` | 26 | 0.830 | 0.580 | 0.572 | 0.294 | 0.130 | 0.445 | 0.910 |
| `mazes_seed2` | 27 | 0.790 | 0.580 | 0.563 | 0.308 | 0.108 | 0.481 | 0.900 |

## Is the planner equally fragile, model to model?

From each model's own exhaustive sweep of all ~102 000 single-cell alterations.

| model | maps with a damaging cell | median damaging cells | worst Opt |
|-------|---------------------------|-----------------------|-----------|
| `pretrained` | 85.5% | 0.78% | 1.353 |
| `mazes_seed0` | 87.5% | 0.78% | 1.310 |
| `mazes_seed1` | 81.9% | 0.73% | 1.370 |
| `mazes_seed2` | 81.0% | 0.59% | 1.350 |


## The same alterations, decomposed by every model

Repeating step 7 per model uses each model's own worst flips, which mixes two sources of variance: the models differ and the alterations differ. Here every model is run on one fixed set — the flips chosen as worst for the shipped checkpoint — which isolates the model difference.

| model | n damaged | path: encoder | path: search | n damaged | effort: encoder | effort: search |
|-------|-----------|---------------|--------------|-----------|-----------------|----------------|
| `pretrained` | 70 | +0.493 | +0.284 | 55 | +0.229 | +0.256 |
| `mazes_seed0` | 36 | +0.461 | +0.639 | 44 | +0.276 | +0.385 |
| `mazes_seed1` | 34 | +0.559 | +0.554 | 48 | +0.487 | +0.338 |
| `mazes_seed2` | 40 | +0.465 | +0.461 | 39 | +0.316 | +0.601 |

**Read the `n` columns before the fractions.** The alterations are the same, but how many of them actually damage each model is not: a flip chosen as worst for one model often leaves another untouched. So the fractions are still averages over different subsets, and only the comparison of *which route dominates* is meaningful — not the exact values.


## Mediation per model (clean failure mode only)

Each model has its own worst flips, so each needs its own sweep and swap. Restricted to alterations that did not change the map's optimum.

| model | path: encoder | path: search | effort: encoder | effort: search |
|-------|---------------|--------------|-----------------|----------------|
| `pretrained` | +0.493 | +0.284 | +0.229 | +0.256 |
| `mazes_seed0` | +0.463 | +0.437 | +0.150 | +0.327 |
| `mazes_seed1` | +0.598 | +0.304 | +0.160 | +0.298 |
| `mazes_seed2` | +0.627 | +0.292 | +0.040 | +0.363 |

## Do the models agree on the cost map itself?

The cost map is the object this project interprets. Correlations over free, reachable cells; the within-map column removes any contribution from models simply disagreeing about which maps are hard.

| pair | pooled r | within-map r | mean \|difference\| |
|------|----------|--------------|---------------------|
| pretrained vs mazes_seed0 | 0.848 | 0.842 | 0.114 |
| pretrained vs mazes_seed1 | 0.801 | 0.792 | 0.126 |
| pretrained vs mazes_seed2 | 0.841 | 0.838 | 0.116 |
| mazes_seed0 vs mazes_seed1 | 0.870 | 0.866 | 0.100 |
| mazes_seed0 vs mazes_seed2 | 0.884 | 0.881 | 0.099 |
| mazes_seed1 vs mazes_seed2 | 0.833 | 0.829 | 0.119 |

## Do the models break on the same cells?

Every model's sweep labels all ~102 000 single-cell alterations as damaging or not. Comparing the labels separates a property of the map — this cell is dangerous for any planner — from a property of the model.

| pair | damaging for each | damaging for both | share | vs chance | same worst flip |
|------|-------------------|-------------------|-------|-----------|-----------------|
| pretrained vs mazes_seed0 | 2,040 / 2,188 | 920 | 45% | **21x** | 32/100 maps |
| pretrained vs mazes_seed1 | 2,040 / 1,949 | 1,007 | 49% | **26x** | 30/100 maps |
| pretrained vs mazes_seed2 | 2,040 / 1,898 | 1,037 | 51% | **27x** | 37/100 maps |
| mazes_seed0 vs mazes_seed1 | 2,188 / 1,949 | 797 | 36% | **19x** | 34/100 maps |
| mazes_seed0 vs mazes_seed2 | 2,188 / 1,898 | 802 | 37% | **20x** | 32/100 maps |
| mazes_seed1 vs mazes_seed2 | 1,949 / 1,898 | 976 | 50% | **27x** | 33/100 maps |


## Dispersion

| quantity | mean | std | min | max |
|---|---|---|---|---|
| `checkpoint_epoch` | 27.2500 | 4.1932 | 23.0000 | 33.0000 |
| `planning.p_opt` | 0.8125 | 0.0206 | 0.7900 | 0.8300 |
| `planning.mean_opt` | 1.0174 | 0.0018 | 1.0154 | 1.0195 |
| `planning.mean_exp` | 0.5856 | 0.0127 | 0.5779 | 0.6046 |
| `planning.h_mean` | 0.5651 | 0.0093 | 0.5530 | 0.5728 |
| `surrogate_r2.geometric_h1_gbm` | 0.2790 | 0.0294 | 0.2403 | 0.3080 |
| `surrogate_r2.path_gbm` | 0.1200 | 0.0092 | 0.1082 | 0.1301 |
| `surrogate_r2.all_gbm` | 0.4427 | 0.0297 | 0.4083 | 0.4805 |
| `sufficiency_all.p_opt` | 0.9150 | 0.0129 | 0.9000 | 0.9300 |
| `sufficiency_all.mean_opt` | 1.0072 | 0.0014 | 1.0060 | 1.0091 |
| `sufficiency_all.mean_exp` | 0.6183 | 0.0089 | 0.6059 | 0.6269 |
| `sufficiency_all.h_mean` | 0.5542 | 0.0015 | 0.5521 | 0.5553 |
| `mediation_optimality.n` | 69.0000 | 1.4142 | 67.0000 | 70.0000 |
| `mediation_optimality.encoder` | 0.5453 | 0.0796 | 0.4631 | 0.6273 |
| `mediation_optimality.search` | 0.3293 | 0.0721 | 0.2843 | 0.4367 |
| `mediation_efficiency.n` | 54.5000 | 3.8730 | 49.0000 | 58.0000 |
| `mediation_efficiency.encoder` | 0.1447 | 0.0784 | 0.0396 | 0.2291 |
| `mediation_efficiency.search` | 0.3110 | 0.0457 | 0.2556 | 0.3633 |
| `fragility.maps_with_a_damaging_cell` | 0.8400 | 0.0305 | 0.8101 | 0.8750 |
| `fragility.median_damaging_cells` | 0.0072 | 0.0009 | 0.0059 | 0.0078 |
| `fragility.mean_damaging_cells` | 0.0198 | 0.0012 | 0.0186 | 0.0214 |
| `fragility.worst_opt` | 1.3459 | 0.0254 | 1.3103 | 1.3704 |
| `fragility.median_worst_opt` | 1.0620 | 0.0038 | 1.0588 | 1.0667 |
| `shared_optimality.n` | 45.0000 | 16.8523 | 34.0000 | 70.0000 |
| `shared_optimality.encoder` | 0.4944 | 0.0452 | 0.4606 | 0.5588 |
| `shared_optimality.search` | 0.4845 | 0.1518 | 0.2843 | 0.6386 |
| `shared_efficiency.n` | 46.5000 | 6.7577 | 39.0000 | 55.0000 |
| `shared_efficiency.encoder` | 0.3269 | 0.1124 | 0.2291 | 0.4870 |
| `shared_efficiency.search` | 0.3949 | 0.1473 | 0.2556 | 0.6007 |

