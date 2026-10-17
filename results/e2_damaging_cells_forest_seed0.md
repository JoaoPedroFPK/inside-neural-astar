# Where the damaging cells are (exhaustive k=1 sweep)

All 102,200 single-cell flips of the 100 test maps, joined against the step-4 feature library on the flipped cell. 1,895 of them (1.85%) make the planner worse than its own baseline on that map.

Because this covers every cell rather than a sample, it supersedes the direction estimates from the step-3 random pilot, which rested on a few dozen draws.

## Does the *size* of the cost-map change predict damage?

The project's hypothesis says large distortions steer the search to a worse path while small ones only inflate the node count. The manifest records the change in the encoder's output for every flip, so this is directly testable.

* Correlation of ΔCost with *whether* the flip damaged the path: **+0.114**.
* Among damaged flips, correlation with *how much* worse: **+0.081**.
* Mean ΔCost, damaging 10.81 against intact 7.54.

| ΔCost bin | cells | mean ΔCost | damage rate | damage rate, on-path cells only |
|---|---|---|---|---|
| 1 | 20,440 | 3.03 | 0.5% | 3.4% |
| 2 | 20,440 | 4.97 | 0.8% | 5.6% |
| 3 | 20,440 | 6.83 | 1.2% | 5.1% |
| 4 | 20,440 | 9.36 | 2.1% | 5.9% |
| 5 | 20,440 | 13.79 | 4.6% | — |

The relationship is weak and **not monotone**: the damage rate rises across the first four bins and then falls for the largest distortions, and it does so even when restricted to cells on an optimal route, which controls for *where* the flip lands. The size of the distortion is not the operative variable; its location is.

## By what the flip does

| group | cells | damage rate |
|---|---|---|
| free -> wall (pillar) | 84,677 | **1.99%** |
| wall -> free (door) | 17,523 | **1.18%** |

## By whether the cell lies on an optimal route

| group | cells | damage rate |
|---|---|---|
| on some optimal path | 13,151 | **5.03%** |
| off every optimal path | 89,049 | **1.39%** |

## By distance to the nearest wall

| group | cells | damage rate |
|---|---|---|
| obstacle | 17,523 | **1.18%** |
| wall_adjacent | 28,598 | **1.30%** |
| near_wall | 24,621 | **1.65%** |
| interior | 31,458 | **2.90%** |

## Free cells, crossed

| group | cells | damage rate |
|---|---|---|
| on path, wall_adjacent | 3,092 | **5.95%** |
| on path, near_wall | 3,781 | **4.18%** |
| on path, interior | 6,278 | **5.08%** |
| off path, wall_adjacent | 25,506 | **0.73%** |
| off path, near_wall | 20,840 | **1.19%** |
| off path, interior | 25,180 | **2.35%** |

