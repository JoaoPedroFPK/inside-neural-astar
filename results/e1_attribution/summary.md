# Step 8 — attribution of single cost-map values

20 curated cells, integrated gradients with 64 steps (19 s). Figures: `attribution_attacked.png`, `attribution_clean.png`, `attribution_wall_adjacent.png`, `attribution_interior.png`.

## Method checks

* **Receptive field.** Attribution mass must lie within radius 5 of the explained cell, since the encoder is five 3×3 convolutions and cannot be influenced from further away. Observed maximum radius over all cells: **5**; minimum share of mass within radius 5: **1.000**.
* **Completeness.** Integrated gradients should sum to the change in the explained value between baseline and input; mean absolute error **5.38e-03**.
* **Randomisation (Adebayo et al., 2018).** Re-running the attribution with randomised encoder weights should change it. Mean correlation between the trained and randomised attributions: **+0.042** (max |r| 0.691, mean Spearman +0.052). The absolute-value cosine is higher (0.659), which is expected: the receptive field constrains *where* mass can land regardless of training.

## By group

| group | mean \|attribution\| | obstacles within radius 5 | per nearby obstacle | randomisation r |
|-------|---------------------|---------------------------|---------------------|-----------------|
| attacked | 3.925 | 16.0 | 0.249 | -0.138 |
| clean | 2.233 | 13.2 | 0.149 | -0.110 |
| interior | 0.552 | 6.0 | 0.097 | -0.054 |
| wall_adjacent | 4.544 | 17.8 | 0.286 | +0.334 |

**The totals need a control.** With the `free` baseline the input difference is non-zero only at obstacle cells, so total attribution mass scales with local clutter: across the curated set it correlates 0.82 with the number of obstacles inside the receptive field. The per-obstacle column divides that out. Attacked cells keep a higher attribution density than clean ones after the correction, which is the comparison worth quoting - on five cells per group, so it is a direction, not a measurement.

## Which methods respect the receptive field

The encoder is five 3x3 convolutions, so an input cell further than 5 away from the explained cell **cannot** influence it. That makes a hard test most settings cannot run, because the true influence region is usually unknown. Here it is exact:

| method | share of mass within radius 5 | mean furthest non-zero cell |
|--------|-------------------------------|------------------------------|
| grad_cam | 10.9% | 24.6 |
| integrated_gradients | 100.0% | 3.8 |
| saliency | 100.0% | 5.0 |

Integrated gradients and plain saliency place **all** their mass inside the field. Grad-CAM does not come close: it spreads most of its mass over regions that provably had no influence on the value being explained. The reason is structural - Grad-CAM weights each feature channel by that channel's *mean* gradient over the whole map, so a channel that matters anywhere lights up everywhere it is active. On an architecture with no downsampling, that produces a picture of where the network is active in general, not of what produced this number.


Per-cell values: `cells.csv`.

