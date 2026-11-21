# Step 13 — adaptive attack on step 12's detector (mazes_seed1)

Detector: convolutional map-only control + residual statistics, fitted on 50 maps (clean vs. worst efficiency flip), never refitted. On each of the 50 held-out maps the attacker plays the single flip the detector scores least altered among those keeping at least a fraction `f` of the worst flip's Exp damage. `f = 1` is step 12's attack.

`single`: any one flip. `pair`: step 12's flip plus one more cell anywhere, chosen only to hide it. Damage kept is the Exp increase over the clean map, divided by that of step 12's flip. Held-out maps where no flip raises Exp are left out (nothing to hide).

| attacker | f | maps | control AUC | control + residual | gap, 95 % CI | unpaired gap, 95 % CI | damage kept (mean / min) | not step 12's alteration | candidates per map |
|---|---|---|---|---|---|---|---|---|---|
| single | 1.00 | 46 | 0.762 | 0.894 | +0.132 [+0.055, +0.202] | +0.100 [-0.031, +0.248] | 1.00 / 1.00 | 13% | 1.4 |
| single | 0.90 | 46 | 0.752 | 0.880 | +0.128 [+0.053, +0.201] | +0.108 [-0.031, +0.261] | 0.99 / 0.90 | 30% | 2.0 |
| single | 0.75 | 46 | 0.745 | 0.880 | +0.135 [+0.059, +0.209] | +0.112 [-0.028, +0.264] | 0.97 / 0.75 | 39% | 3.1 |
| single | 0.50 | 46 | 0.737 | 0.847 | +0.110 [+0.032, +0.182] | +0.123 [-0.035, +0.294] | 0.88 / 0.52 | 50% | 6.1 |
| pair | 1.00 | 46 | 0.763 | 0.877 | +0.114 [+0.028, +0.187] | +0.093 [-0.044, +0.240] | 1.07 / 1.00 | 100% | 233.8 |
| pair | 0.90 | 46 | 0.753 | 0.874 | +0.121 [+0.032, +0.198] | +0.108 [-0.035, +0.265] | 1.04 / 0.91 | 100% | 247.9 |
| pair | 0.75 | 46 | 0.758 | 0.874 | +0.117 [+0.031, +0.191] | +0.095 [-0.044, +0.244] | 1.03 / 0.83 | 100% | 252.9 |
| pair | 0.50 | 46 | 0.751 | 0.871 | +0.119 [+0.032, +0.195] | +0.104 [-0.037, +0.259] | 1.00 / 0.51 | 100% | 256.0 |
