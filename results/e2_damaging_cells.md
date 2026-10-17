# Where the damaging cells are (exhaustive k=1 sweep)

All 102,198 single-cell flips of the 100 test maps, joined against the step-4 feature library on the flipped cell. 2,040 of them (2.00%) make the planner worse than its own baseline on that map.

Because this covers every cell rather than a sample, it supersedes the direction estimates from the step-3 random pilot, which rested on a few dozen draws.

## Does the *size* of the cost-map change predict damage?

The project's hypothesis says large distortions steer the search to a worse path while small ones only inflate the node count. The manifest records the change in the encoder's output for every flip, so this is directly testable.

* Correlation of ΔCost with *whether* the flip damaged the path: **+0.044**.
* Among damaged flips, correlation with *how much* worse: **-0.061**.
* Mean ΔCost, damaging 12.80 against intact 11.14.

| ΔCost bin | cells | mean ΔCost | damage rate | damage rate, on-path cells only |
|---|---|---|---|---|
| 1 | 20,440 | 5.13 | 0.7% | 3.4% |
| 2 | 20,439 | 7.80 | 1.7% | 5.8% |
| 3 | 20,440 | 10.31 | 2.7% | 8.5% |
| 4 | 20,439 | 13.44 | 2.9% | 3.7% |
| 5 | 20,440 | 19.20 | 2.1% | — |

The relationship is weak and **not monotone**: the damage rate rises across the first four bins and then falls for the largest distortions, and it does so even when restricted to cells on an optimal route, which controls for *where* the flip lands. The size of the distortion is not the operative variable; its location is.

## By what the flip does

| group | cells | damage rate |
|---|---|---|
| free -> wall (pillar) | 91,121 | **1.83%** |
| wall -> free (door) | 11,077 | **3.38%** |

## By whether the cell lies on an optimal route

| group | cells | damage rate |
|---|---|---|
| on some optimal path | 11,997 | **5.34%** |
| off every optimal path | 90,201 | **1.55%** |

## By distance to the nearest wall

| group | cells | damage rate |
|---|---|---|
| obstacle | 11,077 | **3.38%** |
| wall_adjacent | 29,310 | **1.26%** |
| near_wall | 23,835 | **1.68%** |
| interior | 37,976 | **2.36%** |

## Free cells, crossed

| group | cells | damage rate |
|---|---|---|
| on path, wall_adjacent | 2,534 | **3.47%** |
| on path, near_wall | 2,857 | **5.11%** |
| on path, interior | 6,606 | **6.16%** |
| off path, wall_adjacent | 26,776 | **1.05%** |
| off path, near_wall | 20,978 | **1.21%** |
| off path, interior | 31,370 | **1.56%** |

