# Step 9 — larger budgets: greedy search and mediation vs budget

Split `test`, seed 0, 100 maps, candidate pool 64 cells per round, 30 min.

Greedy search adds one cell at a time, re-evaluating every candidate against the set already chosen. Two approximations, both deliberate: it cannot find a pair of cells that only hurts jointly, and its candidate pool is the top of the exhaustive k=1 ranking rather than the whole grid. The random draws of step 3 are the control for both.

## Optimality objective

| budget | greedy mean Opt | greedy worst Opt | random mean Opt |
|--------|---------------------|-----------------------|----------------------|
| 1 | 1.1100 | 1.3529 | 1.0173 |
| 2 | 1.1370 | 1.5000 | 1.0169 |
| 4 | 1.1564 | 1.5652 | 1.0175 |
| 8 | 1.1669 | 1.5652 | 1.0201 |

| budget | encoder | search | encoder (optimum unchanged) | search (optimum unchanged) | flips that opened a shortcut | interaction |
|--------|---------|--------|-----------------------------|----------------------------|------------------------------|-------------|
| 1 | +0.40 | +0.40 | +0.49 | +0.28 | 18% | 24% |
| 2 | +0.35 | +0.39 | +0.43 | +0.36 | 25% | 40% |
| 4 | +0.34 | +0.40 | +0.42 | +0.39 | 26% | 44% |
| 8 | +0.32 | +0.41 | +0.38 | +0.42 | 27% | 45% |

## Efficiency objective

| budget | greedy mean Exp | greedy worst Exp | random mean Exp |
|--------|---------------------|-----------------------|----------------------|
| 1 | 1.1516 | 4.5517 | 0.5869 |
| 2 | 1.4521 | 6.2857 | 0.5808 |
| 4 | 1.7559 | 7.2857 | 0.5960 |
| 8 | 1.9812 | 7.4286 | 0.5988 |

| budget | encoder | search | encoder (optimum unchanged) | search (optimum unchanged) | flips that opened a shortcut | interaction |
|--------|---------|--------|-----------------------------|----------------------------|------------------------------|-------------|
| 1 | +0.19 | +0.50 | +0.23 | +0.26 | 38% | 51% |
| 2 | +0.22 | +0.53 | +0.25 | +0.27 | 53% | 52% |
| 4 | +0.24 | +0.48 | +0.24 | +0.29 | 62% | 57% |
| 8 | +0.25 | +0.48 | +0.23 | +0.34 | 69% | 68% |

Plot: `e3_budget.png`.

