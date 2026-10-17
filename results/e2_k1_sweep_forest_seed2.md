# Step 6 — exhaustive single-cell flip sweep

Split `test`, seed 0: every admissible single-cell flip of every map — 102,200 alterations over 100 maps, 35.0 min (20.6 ms each).

Each alteration is scored against the **altered** map's own optimum and classical-A\* effort, so a planner that correctly re-plans after the change is not counted as failing.

## How fragile is the planner to a single targeted cell?

- **85%** of the 53 maps that were optimal before the flip have at least one cell whose flip costs optimality.
- But the damaging cells are rare: a median of **0.88%** of candidate cells per map (mean 2.40%).
- Worst single-flip damage anywhere in the split: Opt ratio **1.393**; median across maps 1.067.

## The ten most fragile maps

| instance | baseline Opt | worst Opt | cell | damaging cells | worst Exp (Opt ≤ 1.05) | cell |
|----------|--------------|-----------|------|----------------|------------------------|------|
| 39 | 1.036 | **1.393** | (21, 5) | 4.7% | 0.582 | (18, 9) |
| 13 | 1.105 | **1.368** | (14, 10) | 13.1% | 0.778 | (23, 19) |
| 99 | 1.129 | **1.355** | (13, 18) | 10.6% | 0.884 | (11, 17) |
| 89 | 1.192 | **1.346** | (16, 23) | 3.5% | 3.036 | (14, 13) |
| 35 | 1.273 | **1.333** | (21, 20) | 0.9% | nan | — |
| 6 | 1.000 | **1.286** | (17, 18) | 1.4% | 1.167 | (17, 12) |
| 0 | 1.241 | **1.276** | (11, 18) | 1.0% | 1.867 | (18, 15) |
| 5 | 1.000 | **1.273** | (10, 29) | 0.3% | 0.211 | (21, 21) |
| 53 | 1.174 | **1.261** | (9, 15) | 11.2% | 1.042 | (9, 8) |
| 67 | 1.217 | **1.261** | (24, 21) | 0.1% | 1.750 | (16, 10) |

## Selected flips

Two alterations are tagged per map in `results/manifest.parquet`: the worst by Opt ratio (`is_worst_optimality`), and the worst by Exp ratio among alterations costing at most 1.05 in Opt (`is_worst_efficiency`). Step 7 runs the 2×2 swap on these.

Manifest columns follow the project schema; the swap columns (`opt_C0O1`, `opt_C1O0`, `exp_C0O1`, `exp_C1O0`) are added by step 7.

