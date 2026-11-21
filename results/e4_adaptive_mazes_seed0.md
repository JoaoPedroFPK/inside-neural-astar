# Step 13 — adaptive attack on step 12's detector (mazes_seed0)

Detector: convolutional map-only control + residual statistics, fitted on 50 maps (clean vs. worst efficiency flip), never refitted. On each of the 50 held-out maps the attacker plays the single flip the detector scores least altered among those keeping at least a fraction `f` of the worst flip's Exp damage. `f = 1` is step 12's attack.

`single`: any one flip. `pair`: step 12's flip plus one more cell anywhere, chosen only to hide it. Damage kept is the Exp increase over the clean map, divided by that of step 12's flip. Held-out maps where no flip raises Exp are left out (nothing to hide).

| attacker | f | maps | control AUC | control + residual | gap, 95 % CI | unpaired gap, 95 % CI | damage kept (mean / min) | not step 12's alteration | candidates per map |
|---|---|---|---|---|---|---|---|---|---|
| single | 1.00 | 47 | 0.800 | 0.893 | +0.093 [+0.027, +0.163] | +0.038 [-0.066, +0.152] | 1.00 / 1.00 | 0% | 1.2 |
| single | 0.90 | 47 | 0.788 | 0.879 | +0.091 [+0.026, +0.165] | +0.047 [-0.059, +0.165] | 0.98 / 0.91 | 21% | 1.9 |
| single | 0.75 | 47 | 0.779 | 0.854 | +0.075 [+0.010, +0.139] | +0.034 [-0.082, +0.159] | 0.94 / 0.76 | 38% | 3.7 |
| single | 0.50 | 47 | 0.710 | 0.807 | +0.097 [+0.024, +0.166] | +0.069 [-0.071, +0.209] | 0.85 / 0.50 | 53% | 8.8 |
| pair | 1.00 | 47 | 0.759 | 0.863 | +0.104 [+0.026, +0.184] | +0.053 [-0.098, +0.201] | 1.04 / 1.00 | 100% | 232.9 |
| pair | 0.90 | 47 | 0.769 | 0.861 | +0.093 [+0.014, +0.172] | +0.033 [-0.112, +0.180] | 1.02 / 0.91 | 100% | 247.8 |
| pair | 0.75 | 47 | 0.767 | 0.857 | +0.090 [+0.014, +0.171] | +0.022 [-0.133, +0.178] | 1.01 / 0.76 | 100% | 252.7 |
| pair | 0.50 | 47 | 0.763 | 0.842 | +0.080 [+0.007, +0.153] | -0.002 [-0.147, +0.140] | 0.96 / 0.50 | 100% | 256.0 |
