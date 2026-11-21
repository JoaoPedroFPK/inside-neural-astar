# Step 13 — adaptive attack on step 12's detector (mazes_seed2)

Detector: convolutional map-only control + residual statistics, fitted on 50 maps (clean vs. worst efficiency flip), never refitted. On each of the 50 held-out maps the attacker plays the single flip the detector scores least altered among those keeping at least a fraction `f` of the worst flip's Exp damage. `f = 1` is step 12's attack.

`single`: any one flip. `pair`: step 12's flip plus one more cell anywhere, chosen only to hide it. Damage kept is the Exp increase over the clean map, divided by that of step 12's flip. Held-out maps where no flip raises Exp are left out (nothing to hide).

| attacker | f | maps | control AUC | control + residual | gap, 95 % CI | unpaired gap, 95 % CI | damage kept (mean / min) | not step 12's alteration | candidates per map |
|---|---|---|---|---|---|---|---|---|---|
| single | 1.00 | 46 | 0.780 | 0.863 | +0.083 [+0.009, +0.150] | +0.110 [-0.033, +0.260] | 1.00 / 1.00 | 11% | 1.5 |
| single | 0.90 | 46 | 0.756 | 0.845 | +0.089 [+0.019, +0.157] | +0.115 [-0.039, +0.277] | 0.99 / 0.90 | 33% | 2.2 |
| single | 0.75 | 46 | 0.736 | 0.826 | +0.090 [+0.015, +0.165] | +0.087 [-0.076, +0.254] | 0.93 / 0.77 | 57% | 3.5 |
| single | 0.50 | 46 | 0.748 | 0.811 | +0.063 [-0.011, +0.134] | +0.076 [-0.087, +0.242] | 0.89 / 0.50 | 59% | 7.3 |
| pair | 1.00 | 46 | 0.805 | 0.809 | +0.004 [-0.079, +0.086] | +0.026 [-0.102, +0.152] | 1.24 / 1.00 | 100% | 234.1 |
| pair | 0.90 | 46 | 0.789 | 0.793 | +0.004 [-0.080, +0.085] | +0.038 [-0.094, +0.165] | 1.24 / 0.90 | 100% | 249.2 |
| pair | 0.75 | 46 | 0.782 | 0.787 | +0.006 [-0.080, +0.092] | +0.038 [-0.094, +0.165] | 1.22 / 0.79 | 100% | 253.3 |
| pair | 0.50 | 46 | 0.798 | 0.785 | -0.013 [-0.095, +0.073] | +0.011 [-0.108, +0.129] | 1.19 / 0.52 | 100% | 256.0 |
