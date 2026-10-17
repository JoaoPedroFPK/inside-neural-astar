# Step 3 — random-flip pilot

Split `test` (100 instances), seed 0, 3 random draws per budget, 1200 alterations evaluated in 20.8 s (cpu).

Flips are drawn uniformly from all cells except the start and the goal (so a flip may add *or* remove an obstacle), and any draw that disconnects the goal is redrawn. `Opt` and `Exp` are measured against the **altered** map's optimum and classical A\* expansion count — classical A\* replanning after a flip is the correct response, not an error.

## Fragility by budget

| k | n | Opt > 1 | newly suboptimal | Exp > 1.5 | newly Exp > 1.5 | mean Opt | mean Exp |
|---|---|---------|------------------|-----------|-----------------|----------|----------|
| 1 | 300 | 18% | 1% | 1% | 1% | 1.0173 | 0.5869 |
| 2 | 300 | 19% | 3% | 1% | 1% | 1.0169 | 0.5808 |
| 4 | 300 | 20% | 4% | 1% | 1% | 1.0175 | 0.5960 |
| 8 | 300 | 22% | 9% | 1% | 1% | 1.0201 | 0.5988 |

## What the flip did to the map

| k | optimum changed | optimum got shorter | mean Opt when newly broken | worst |
|---|-----------------|---------------------|----------------------------|-------|
| 1 | 0.0% | 0.0% | 1.1206 | 1.2381 |
| 2 | 2.0% | 1.0% | 1.0755 | 1.2333 |
| 4 | 4.3% | 3.0% | 1.0698 | 1.2222 |
| 8 | 8.3% | 3.7% | 1.0810 | 1.3333 |

At k=1 the optimal cost changes in only 0% of draws, so every k=1 failure is the planner deviating on a map whose optimum it already matched — no "the map genuinely changed" confound. Damage, when it happens, is not marginal: the newly-broken alterations average 1.121 and reach 1.333 across budgets (a third longer than the optimum).

## Pillars vs doors (k=1)

| flip direction | n | share of draws | newly suboptimal | mean Opt | mean ΔCost L1 |
|----------------|---|----------------|------------------|----------|---------------|
| free->obstacle (pillar) | 275 | 91.7% | 0.9% | 1.0176 | 10.419 |
| obstacle->free (door) | 25 | 8.3% | 4.5% | 1.0132 | 12.958 |

The maps are ~11% obstacles, with one-cell-thick walls separating open areas roughly ten cells wide (`docs/verified.md` § 11). So the two flip directions are different edits: making a free cell an obstacle drops an isolated **pillar** into open space, which an 8-connected search walks around at no cost, while making an obstacle free opens a **door** between two areas. Doors are rarer (they are only 11% of cells) but several times more damaging per flip — which is the main reason uniform random flipping is such a weak attack.

* **Opt > 1** — share of alterations where Neural A\*'s path is longer than the altered map's optimum (PLAN.md's headline number).
* **newly suboptimal** — the same share, restricted to alterations whose unaltered baseline was already optimal. This is the damage *caused* by the flip, separated from the planner's standing error rate (baseline mean Opt 1.0154, mean Exp 0.5779).
* **Exp > 1.5** — share expanding more than 1.5x classical A\* on the altered map.

## ΔCost in the encoder output

| k | mean ΔCost L1 | mean share of ΔCost mass under the flipped cells |
|---|---------------|---------------------------------------------------|
| 1 | 10.630 | 1.7% |
| 2 | 21.681 | 1.7% |
| 4 | 40.486 | 1.7% |
| 8 | 72.266 | 2.0% |

A flipped cell changes the cost map over a neighbourhood, not only at the flipped cell: with k=1 only 1.7% of the total |Δcost| mass sits on the flipped cell itself. The spread is bounded exactly: the encoder is five 3x3 convolutions, so its receptive field has radius 5, and |Δcost| from a k=1 flip is *exactly zero* outside the 11x11 window around the flipped cell (verified in docs/verified.md section 10). Any behaviour change further away travelled through the search, not through the encoder's view of that region.

## Decision: which budgets deserve a targeted search

Random flips are a weak attack. At k=1 only 1% of alterations push a previously-optimal instance off the optimum, rising to 9% at k=8; the efficiency failure rate stays at 1% across all budgets. Most of the raw `Opt > 1` column is the planner's standing error rate, not flip damage.

The exhaustive k=1 probe on 3 instances (3066 alterations, 33 s) says it is a weak baseline, not a robust planner:

| instance | baseline Opt | worst Opt from one flip | cell | share of cells that damage Opt | worst Exp (Opt ≤ 1.05) |
|----------|--------------|-------------------------|------|--------------------------------|------------------------|
| 0 | 1.000 | **1.053** | (14, 17) | 0.1% | 0.796 |
| 1 | 1.000 | **1.120** | (24, 10) | 0.1% | 0.344 |
| 2 | 1.000 | **1.111** | (20, 14) | 1.2% | 3.750 |

Every one of the 3/3 instances has at least one single cell whose flip costs optimality (worst `Opt` 1.053–1.120), but only 0.1%–1.2% of the ~1 000 candidate cells do — one to a dozen cells per map. A random draw lands on one with probability ≈0.5% per flip, the same order as the 1% damage rate the pilot measured at k=1 — consistent with the two measuring the same thing, though the probe covers only three maps.

**Fragility is real but extremely localized.** That is a substantive result for the project, not just a methodological note: it means the mediation question of E3 has to be asked about *targeted* flips, because the average flip carries almost no damage to decompose.

Consequently:

* **k=1 exhaustive (step 6): worth it, and necessary.** At 11 ms per alteration the full test split costs ≈18 min single-process. Random search cannot substitute: it would need ~100–1000 draws per map to hit the same cells.
* **k ∈ {2, 4, 8} greedy (step 9): worth it.** Random damage does grow with budget (1% → 9%), so there is signal to climb, and greedy starts from the k=1 cells this sweep will have found.
* **Random flips stay as the control** at every budget, which is what PLAN.md § 5 asks for to keep the attack-selection bias visible.
* **The efficiency objective needs watching.** Only 1/3 probe instances have any cell pushing `Exp` above 1.5 (one reaches 3.75). If the full sweep confirms that, the efficiency arm of E2 will have a small sample and the H2 decomposition may only be answerable for the optimality objective.

