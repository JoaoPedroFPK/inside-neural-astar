# Step 6 — exhaustive single-cell flip sweep

Split `test`, seed 0: every admissible single-cell flip of every map — 102,198 alterations over 100 maps, 25.9 min (15.2 ms each).

Each alteration is scored against the **altered** map's own optimum and classical-A\* effort, so a planner that correctly re-plans after the change is not counted as failing.

## How fragile is the planner to a single targeted cell?

- **81%** of the 79 maps that were optimal before the flip have at least one cell whose flip costs optimality.
- But the damaging cells are rare: a median of **0.59%** of candidate cells per map (mean 1.86%).
- Worst single-flip damage anywhere in the split: Opt ratio **1.350**; median across maps 1.059.

## The ten most fragile maps

| instance | baseline Opt | worst Opt | cell | damaging cells | worst Exp (Opt ≤ 1.05) | cell |
|----------|--------------|-----------|------|----------------|------------------------|------|
| 70 | 1.150 | **1.350** | (17, 13) | 1.8% | 1.048 | (18, 12) |
| 95 | 1.000 | **1.333** | (10, 24) | 8.0% | 4.036 | (17, 21) |
| 36 | 1.179 | **1.321** | (11, 27) | 1.2% | 1.069 | (19, 17) |
| 8 | 1.000 | **1.310** | (20, 5) | 8.2% | 3.600 | (8, 21) |
| 35 | 1.000 | **1.300** | (18, 3) | 4.8% | 0.446 | (12, 10) |
| 51 | 1.000 | **1.294** | (18, 10) | 0.9% | 0.543 | (14, 20) |
| 62 | 1.294 | **1.294** | (0, 0) | 0.0% | 1.111 | (11, 12) |
| 86 | 1.120 | **1.280** | (13, 27) | 1.6% | 1.154 | (23, 11) |
| 23 | 1.185 | **1.259** | (11, 21) | 1.4% | 0.431 | (7, 11) |
| 47 | 1.000 | **1.259** | (16, 2) | 2.4% | 3.607 | (15, 10) |

## Selected flips

Two alterations are tagged per map in `results/manifest.parquet`: the worst by Opt ratio (`is_worst_optimality`), and the worst by Exp ratio among alterations costing at most 1.05 in Opt (`is_worst_efficiency`). Step 7 runs the 2×2 swap on these.

Manifest columns follow the project schema; the swap columns (`opt_C0O1`, `opt_C1O0`, `exp_C0O1`, `exp_C1O0`) are added by step 7.

