# Step 6 — exhaustive single-cell flip sweep

Split `test`, seed 0: every admissible single-cell flip of every map — 102,198 alterations over 100 maps, 28.8 min (16.9 ms each).

Each alteration is scored against the **altered** map's own optimum and classical-A\* effort, so a planner that correctly re-plans after the change is not counted as failing.

## How fragile is the planner to a single targeted cell?

- **88%** of the 80 maps that were optimal before the flip have at least one cell whose flip costs optimality.
- But the damaging cells are rare: a median of **0.78%** of candidate cells per map (mean 2.14%).
- Worst single-flip damage anywhere in the split: Opt ratio **1.310**; median across maps 1.064.

## The ten most fragile maps

| instance | baseline Opt | worst Opt | cell | damaging cells | worst Exp (Opt ≤ 1.05) | cell |
|----------|--------------|-----------|------|----------------|------------------------|------|
| 8 | 1.172 | **1.310** | (14, 21) | 1.9% | 1.033 | (15, 9) |
| 70 | 1.150 | **1.300** | (19, 13) | 1.6% | 1.571 | (24, 16) |
| 23 | 1.185 | **1.296** | (23, 6) | 1.2% | 0.417 | (10, 14) |
| 51 | 1.000 | **1.294** | (18, 10) | 2.0% | 0.589 | (8, 9) |
| 62 | 1.000 | **1.294** | (7, 10) | 1.6% | 2.389 | (16, 13) |
| 4 | 1.000 | **1.276** | (12, 21) | 5.2% | 0.515 | (13, 21) |
| 95 | 1.074 | **1.259** | (14, 28) | 4.4% | 4.286 | (16, 21) |
| 99 | 1.000 | **1.257** | (8, 21) | 0.3% | 0.566 | (11, 21) |
| 36 | 1.143 | **1.250** | (11, 27) | 8.1% | 1.034 | (11, 21) |
| 97 | 1.000 | **1.240** | (12, 21) | 1.6% | 1.304 | (23, 10) |

## Selected flips

Two alterations are tagged per map in `results/manifest.parquet`: the worst by Opt ratio (`is_worst_optimality`), and the worst by Exp ratio among alterations costing at most 1.05 in Opt (`is_worst_efficiency`). Step 7 runs the 2×2 swap on these.

Manifest columns follow the project schema; the swap columns (`opt_C0O1`, `opt_C1O0`, `exp_C0O1`, `exp_C1O0`) are added by step 7.

