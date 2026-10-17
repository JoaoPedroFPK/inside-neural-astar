# Step 6 — exhaustive single-cell flip sweep

Split `test`, seed 0: every admissible single-cell flip of every map — 102,198 alterations over 100 maps, 21.1 min (12.4 ms each).

Each alteration is scored against the **altered** map's own optimum and classical-A\* effort, so a planner that correctly re-plans after the change is not counted as failing.

## How fragile is the planner to a single targeted cell?

- **86%** of the 83 maps that were optimal before the flip have at least one cell whose flip costs optimality.
- But the damaging cells are rare: a median of **0.78%** of candidate cells per map (mean 2.00%).
- Worst single-flip damage anywhere in the split: Opt ratio **1.353**; median across maps 1.067.

## The ten most fragile maps

| instance | baseline Opt | worst Opt | cell | damaging cells | worst Exp (Opt ≤ 1.05) | cell |
|----------|--------------|-----------|------|----------------|------------------------|------|
| 20 | 1.000 | **1.353** | (21, 10) | 1.2% | 0.607 | (11, 8) |
| 70 | 1.150 | **1.350** | (19, 12) | 4.3% | 2.619 | (18, 21) |
| 36 | 1.179 | **1.321** | (10, 28) | 1.4% | 1.069 | (13, 21) |
| 35 | 1.000 | **1.300** | (18, 3) | 6.9% | 0.391 | (11, 9) |
| 47 | 1.000 | **1.296** | (16, 1) | 6.6% | 4.321 | (17, 10) |
| 95 | 1.000 | **1.296** | (14, 30) | 6.5% | 4.393 | (16, 21) |
| 51 | 1.000 | **1.294** | (18, 10) | 1.4% | 0.472 | (9, 15) |
| 62 | 1.000 | **1.294** | (14, 15) | 1.6% | 2.278 | (16, 13) |
| 4 | 1.000 | **1.276** | (12, 21) | 3.8% | 0.463 | (15, 21) |
| 8 | 1.138 | **1.276** | (14, 21) | 11.8% | 1.067 | (15, 9) |

## Selected flips

Two alterations are tagged per map in `results/manifest.parquet`: the worst by Opt ratio (`is_worst_optimality`), and the worst by Exp ratio among alterations costing at most 1.05 in Opt (`is_worst_efficiency`). Step 7 runs the 2×2 swap on these.

Manifest columns follow the project schema; the swap columns (`opt_C0O1`, `opt_C1O0`, `exp_C0O1`, `exp_C1O0`) are added by step 7.

