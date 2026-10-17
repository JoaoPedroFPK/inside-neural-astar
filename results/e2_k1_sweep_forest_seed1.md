# Step 6 — exhaustive single-cell flip sweep

Split `test`, seed 0: every admissible single-cell flip of every map — 102,200 alterations over 100 maps, 33.5 min (19.6 ms each).

Each alteration is scored against the **altered** map's own optimum and classical-A\* effort, so a planner that correctly re-plans after the change is not counted as failing.

## How fragile is the planner to a single targeted cell?

- **88%** of the 56 maps that were optimal before the flip have at least one cell whose flip costs optimality.
- But the damaging cells are rare: a median of **0.98%** of candidate cells per map (mean 2.60%).
- Worst single-flip damage anywhere in the split: Opt ratio **1.478**; median across maps 1.056.

## The ten most fragile maps

| instance | baseline Opt | worst Opt | cell | damaging cells | worst Exp (Opt ≤ 1.05) | cell |
|----------|--------------|-----------|------|----------------|------------------------|------|
| 78 | 1.174 | **1.478** | (18, 16) | 5.7% | 1.042 | (7, 8) |
| 13 | 1.263 | **1.474** | (15, 18) | 0.7% | 0.815 | (24, 18) |
| 99 | 1.290 | **1.452** | (1, 9) | 2.3% | nan | — |
| 33 | 1.038 | **1.385** | (14, 11) | 1.8% | 3.037 | (23, 17) |
| 73 | 1.100 | **1.350** | (15, 13) | 7.2% | 1.095 | (19, 20) |
| 35 | 1.212 | **1.303** | (10, 12) | 15.7% | nan | — |
| 6 | 1.000 | **1.286** | (24, 15) | 5.3% | 1.227 | (20, 8) |
| 9 | 1.091 | **1.273** | (11, 15) | 11.1% | 0.423 | (13, 13) |
| 89 | 1.154 | **1.269** | (19, 24) | 5.1% | 3.071 | (19, 18) |
| 67 | 1.217 | **1.261** | (16, 18) | 0.4% | 1.708 | (17, 13) |

## Selected flips

Two alterations are tagged per map in `results/manifest.parquet`: the worst by Opt ratio (`is_worst_optimality`), and the worst by Exp ratio among alterations costing at most 1.05 in Opt (`is_worst_efficiency`). Step 7 runs the 2×2 swap on these.

Manifest columns follow the project schema; the swap columns (`opt_C0O1`, `opt_C1O0`, `exp_C0O1`, `exp_C1O0`) are added by step 7.

