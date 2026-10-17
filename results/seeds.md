# Step 10 — dispersion across training seeds

2 model(s) compared on the same instances and the same fixed starts. A difference between conditions is only a result if it is larger than the difference between seeds.

| model | exactly optimal | mean Exp | Hmean | R² geometry | R² path | R² all | sufficiency Opt |
|-------|-----------------|----------|-------|-------------|---------|--------|-----------------|
| `pretrained` | 0.830 | 0.580 | 0.572 | 0.240 | 0.123 | 0.408 | 0.920 |
| `mazes_seed0` | 0.800 | 0.605 | 0.553 | 0.274 | 0.119 | 0.437 | 0.930 |

## Mediation per model (clean failure mode only)

Each model has its own worst flips, so each needs its own sweep and swap. Restricted to alterations that did not change the map's optimum.

| model | path: encoder | path: search | effort: encoder | effort: search |
|-------|---------------|--------------|-----------------|----------------|
| `pretrained` | +0.493 | +0.284 | +0.229 | +0.256 |

## Do the models agree on the cost map itself?

The cost map is the object this project interprets. Correlations over free, reachable cells; the within-map column removes any contribution from models simply disagreeing about which maps are hard.

| pair | pooled r | within-map r | mean \|difference\| |
|------|----------|--------------|---------------------|
| pretrained vs mazes_seed0 | 0.848 | 0.842 | 0.114 |

## Dispersion

| quantity | mean | std | min | max |
|---|---|---|---|---|
| `planning.p_opt` | 0.8150 | 0.0212 | 0.8000 | 0.8300 |
| `planning.mean_opt` | 1.0174 | 0.0013 | 1.0164 | 1.0183 |
| `planning.mean_exp` | 0.5921 | 0.0177 | 0.5796 | 0.6046 |
| `planning.h_mean` | 0.5625 | 0.0134 | 0.5530 | 0.5720 |
| `surrogate_r2.geometric_h1_gbm` | 0.2570 | 0.0236 | 0.2403 | 0.2738 |
| `surrogate_r2.path_gbm` | 0.1209 | 0.0028 | 0.1189 | 0.1228 |
| `surrogate_r2.all_gbm` | 0.4227 | 0.0203 | 0.4083 | 0.4370 |
| `sufficiency_all.p_opt` | 0.9250 | 0.0071 | 0.9200 | 0.9300 |
| `sufficiency_all.mean_opt` | 1.0067 | 0.0010 | 1.0060 | 1.0074 |
| `sufficiency_all.mean_exp` | 0.6231 | 0.0054 | 0.6193 | 0.6269 |
| `sufficiency_all.h_mean` | 0.5532 | 0.0016 | 0.5521 | 0.5543 |
| `mediation_optimality.n` | 70.0000 | nan | 70.0000 | 70.0000 |
| `mediation_optimality.encoder` | 0.4929 | nan | 0.4929 | 0.4929 |
| `mediation_optimality.search` | 0.2843 | nan | 0.2843 | 0.2843 |
| `mediation_efficiency.n` | 55.0000 | nan | 55.0000 | 55.0000 |
| `mediation_efficiency.encoder` | 0.2291 | nan | 0.2291 | 0.2291 |
| `mediation_efficiency.search` | 0.2556 | nan | 0.2556 | 0.2556 |

