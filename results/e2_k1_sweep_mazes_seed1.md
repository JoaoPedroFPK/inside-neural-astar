# Step 6 — exhaustive single-cell flip sweep

Split `test`, seed 0: every admissible single-cell flip of every map — 102,198 alterations over 100 maps, 27.3 min (16.0 ms each).

Each alteration is scored against the **altered** map's own optimum and classical-A\* effort, so a planner that correctly re-plans after the change is not counted as failing.

## How fragile is the planner to a single targeted cell?

- **82%** of the 83 maps that were optimal before the flip have at least one cell whose flip costs optimality.
- But the damaging cells are rare: a median of **0.73%** of candidate cells per map (mean 1.91%).
- Worst single-flip damage anywhere in the split: Opt ratio **1.370**; median across maps 1.059.

## The ten most fragile maps

| instance | baseline Opt | worst Opt | cell | damaging cells | worst Exp (Opt ≤ 1.05) | cell |
|----------|--------------|-----------|------|----------------|------------------------|------|
| 95 | 1.111 | **1.370** | (11, 21) | 7.4% | 1.071 | (14, 25) |
| 62 | 1.294 | **1.353** | (10, 21) | 0.3% | 1.167 | (11, 12) |
| 8 | 1.138 | **1.310** | (14, 21) | 7.8% | 1.200 | (19, 9) |
| 51 | 1.000 | **1.294** | (17, 10) | 0.8% | 0.511 | (12, 18) |
| 33 | 1.000 | **1.286** | (22, 10) | 0.5% | 1.182 | (17, 10) |
| 36 | 1.143 | **1.286** | (10, 28) | 5.6% | 1.172 | (23, 11) |
| 99 | 1.000 | **1.257** | (8, 21) | 0.5% | 0.609 | (20, 17) |
| 4 | 1.000 | **1.241** | (9, 21) | 5.3% | 0.525 | (13, 21) |
| 54 | 1.207 | **1.241** | (21, 13) | 0.2% | 0.270 | (2, 26) |
| 97 | 1.000 | **1.240** | (12, 21) | 1.5% | 1.217 | (23, 10) |

## Selected flips

Two alterations are tagged per map in `results/manifest.parquet`: the worst by Opt ratio (`is_worst_optimality`), and the worst by Exp ratio among alterations costing at most 1.05 in Opt (`is_worst_efficiency`). Step 7 runs the 2×2 swap on these.

Manifest columns follow the project schema; the swap columns (`opt_C0O1`, `opt_C1O0`, `exp_C0O1`, `exp_C1O0`) are added by step 7.

