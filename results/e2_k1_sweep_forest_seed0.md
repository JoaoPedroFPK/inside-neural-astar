# Step 6 — exhaustive single-cell flip sweep

Split `test`, seed 0: every admissible single-cell flip of every map — 102,200 alterations over 100 maps, 30.5 min (17.9 ms each).

Each alteration is scored against the **altered** map's own optimum and classical-A\* effort, so a planner that correctly re-plans after the change is not counted as failing.

## How fragile is the planner to a single targeted cell?

- **85%** of the 54 maps that were optimal before the flip have at least one cell whose flip costs optimality.
- But the damaging cells are rare: a median of **0.73%** of candidate cells per map (mean 1.85%).
- Worst single-flip damage anywhere in the split: Opt ratio **1.474**; median across maps 1.045.

## The ten most fragile maps

| instance | baseline Opt | worst Opt | cell | damaging cells | worst Exp (Opt ≤ 1.05) | cell |
|----------|--------------|-----------|------|----------------|------------------------|------|
| 13 | 1.368 | **1.474** | (14, 15) | 0.7% | nan | — |
| 99 | 1.161 | **1.452** | (2, 13) | 9.6% | 0.782 | (5, 12) |
| 39 | 1.321 | **1.393** | (8, 16) | 10.7% | 0.538 | (23, 3) |
| 89 | 1.154 | **1.346** | (18, 23) | 7.7% | 2.630 | (22, 6) |
| 35 | 1.152 | **1.303** | (9, 11) | 6.9% | nan | — |
| 73 | 1.150 | **1.300** | (15, 19) | 1.9% | 1.095 | (23, 13) |
| 6 | 1.000 | **1.286** | (23, 12) | 6.9% | 1.227 | (20, 8) |
| 9 | 1.136 | **1.273** | (16, 12) | 5.0% | 0.802 | (16, 10) |
| 38 | 1.208 | **1.250** | (11, 13) | 0.8% | 1.760 | (13, 8) |
| 0 | 1.207 | **1.241** | (16, 19) | 2.2% | 1.900 | (19, 9) |

## Selected flips

Two alterations are tagged per map in `results/manifest.parquet`: the worst by Opt ratio (`is_worst_optimality`), and the worst by Exp ratio among alterations costing at most 1.05 in Opt (`is_worst_efficiency`). Step 7 runs the 2×2 swap on these.

Manifest columns follow the project schema; the swap columns (`opt_C0O1`, `opt_C1O0`, `exp_C0O1`, `exp_C1O0`) are added by step 7.

