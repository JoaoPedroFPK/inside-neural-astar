# Step 7b (part a) — the maze-trained planner across map families

All eight 32x32 families ship with the nested dataset submodule. The planner here is the one trained on `mazes`, so every number below mixes *distribution shift* with *geometry*: it cannot show that a finding generalises, only that it does or does not survive the shift. Part (b) — one encoder trained per family — is what settles generalisation.

## Map structure

| family | obstacles | interior | wall-adjacent | open (topology) | dead ends | single flips that seal the goal |
|--------|-----------|----------|---------------|-----------------|-----------|---------------------------------|
| `mazes` | 10.8% | 37% | 29% | 61% | 0.0% | 0.00% |
| `forest` | 17.1% | 31% | 28% | 58% | 0.0% | 0.00% |
| `single_bugtrap` | 6.2% | 59% | 18% | 77% | 0.0% | 0.00% |
| `multiple_bugtraps` | 9.5% | 41% | 26% | 66% | 0.0% | 0.00% |
| `bugtrap_forest` | 16.3% | 30% | 29% | 57% | 0.0% | 0.00% |
| `alternating_gaps` | 16.9% | 49% | 18% | 66% | 0.0% | 0.00% |
| `shifting_gaps` | 17.0% | 49% | 18% | 66% | 0.0% | 0.00% |
| `gaps_and_forest` | 35.7% | 15% | 29% | 38% | 0.0% | 0.09% |

## Transfer of the maze-trained planner

| family | exactly optimal | mean Opt | mean Exp | Hmean |
|--------|-----------------|----------|----------|-------|
| `mazes` | 0.830 | 1.0154 | 0.578 | 0.573 |
| `forest` | 0.430 | 1.0693 | 0.860 | 0.365 |
| `single_bugtrap` | 0.830 | 1.0183 | 0.920 | 0.287 |
| `multiple_bugtraps` | 0.590 | 1.0370 | 0.842 | 0.373 |
| `bugtrap_forest` | 0.600 | 1.0406 | 1.023 | 0.272 |
| `alternating_gaps` | 0.770 | 1.0461 | 0.680 | 0.490 |
| `shifting_gaps` | 0.770 | 1.0436 | 0.622 | 0.527 |
| `gaps_and_forest` | 0.540 | 1.0411 | 0.646 | 0.435 |

## The pillar/door prediction

On mazes, opening a wall (a *door*) was about five times more damaging per flip than dropping an obstacle into open space (a *pillar*). If that is geometry rather than a property of the planner, families without walls should not show it.

| family | pillar share of draws | pillar damage | door damage | ratio |
|--------|----------------------|---------------|-------------|-------|
| `mazes` | 92% | 0.9% | 4.5% | 5.2x |
| `forest` | 83% | 2.9% | 0.0% | 0.0x |
| `single_bugtrap` | 98% | 0.4% | 0.0% | 0.0x |
| `multiple_bugtraps` | 90% | 1.2% | 6.2% | 5.0x |
| `bugtrap_forest` | 86% | 2.6% | 3.8% | 1.5x |
| `alternating_gaps` | 82% | 1.1% | 2.3% | 2.1x |
| `shifting_gaps` | 82% | 2.7% | 0.0% | 0.0x |
| `gaps_and_forest` | 68% | 1.8% | 2.0% | 1.2x |

