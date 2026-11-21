# Step 13 — adaptive attack on step 12's detector (pretrained)

Detector: convolutional map-only control + residual statistics, fitted on 50 maps (clean vs. worst efficiency flip), never refitted. On each of the 50 held-out maps the attacker plays the single flip the detector scores least altered among those keeping at least a fraction `f` of the worst flip's Exp damage. `f = 1` is step 12's attack.

`single`: any one flip. `pair`: step 12's flip plus one more cell anywhere, chosen only to hide it. Damage kept is the Exp increase over the clean map, divided by that of step 12's flip. Held-out maps where no flip raises Exp are left out (nothing to hide).

| attacker | f | maps | control AUC | control + residual | gap, 95 % CI | unpaired gap, 95 % CI | damage kept (mean / min) | not step 12's alteration | candidates per map |
|---|---|---|---|---|---|---|---|---|---|
| single | 1.00 | 48 | 0.772 | 0.882 | +0.110 [+0.029, +0.191] | +0.170 [+0.029, +0.327] | 1.00 / 1.00 | 4% | 2.1 |
| single | 0.90 | 48 | 0.758 | 0.869 | +0.111 [+0.035, +0.188] | +0.186 [+0.044, +0.343] | 0.99 / 0.91 | 27% | 2.7 |
| single | 0.75 | 48 | 0.752 | 0.853 | +0.101 [+0.018, +0.179] | +0.153 [+0.017, +0.307] | 0.95 / 0.75 | 40% | 3.8 |
| single | 0.50 | 48 | 0.725 | 0.805 | +0.080 [-0.009, +0.165] | +0.156 [+0.011, +0.323] | 0.87 / 0.50 | 52% | 6.6 |
| pair | 1.00 | 48 | 0.775 | 0.871 | +0.096 [+0.006, +0.189] | +0.125 [-0.016, +0.281] | 1.03 / 1.00 | 100% | 233.2 |
| pair | 0.90 | 48 | 0.781 | 0.871 | +0.090 [+0.003, +0.180] | +0.125 [-0.016, +0.281] | 1.01 / 0.91 | 100% | 248.6 |
| pair | 0.75 | 48 | 0.793 | 0.865 | +0.072 [-0.014, +0.166] | +0.125 [-0.016, +0.281] | 0.99 / 0.80 | 100% | 253.8 |
| pair | 0.50 | 48 | 0.793 | 0.863 | +0.071 [-0.016, +0.161] | +0.128 [-0.013, +0.285] | 0.97 / 0.61 | 100% | 256.0 |
