# Where the damaging cells are (exhaustive k=1 sweep)

All 102,198 single-cell flips of the 100 test maps, joined against the step-4 feature library on the flipped cell. 2,040 of them (2.00%) make the planner worse than its own baseline on that map.

Because this covers every cell rather than a sample, it supersedes the direction estimates from the step-3 random pilot, which rested on a few dozen draws.

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

