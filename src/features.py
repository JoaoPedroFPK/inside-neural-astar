"""Geometric feature library for E1 (PLAN.md step 4).

The question E1 asks is what the learned cost map encodes. The answer is tested
by regressing the cost map on interpretable per-cell quantities (step 5), so
this module defines those quantities — once, precisely, with every convention
written down.

Three families, kept separate because step 5 fits them separately:

**Goal geometry** — how far the cell is from the goal, in the three metrics
that matter here: the true geodesic (what the dataset stores), the Euclidean
distance, and the Chebyshev distance (which *is* the upstream heuristic, up to
the 0.001 Euclidean tie-break, so a cost map that merely re-learns `h` would
show up as a near-perfect fit on this one feature).

**Obstacle geometry** — distance to the nearest obstacle, local obstacle
density, and two corridor-width proxies.

**Path indicators** — whether the cell lies on a shortest path. Three variants,
because they are not the same thing and the difference is the whole competing
hypothesis of PLAN.md:

* `on_reference_path` — the dataset's single optimal trajectory from the fixed
  start. This is literally the training target (`L1(histories, opt_trajs)`).
* `on_any_optimal_path` — the union of *all* optimal paths, i.e. cells with
  ``d(start, v) + d(v, goal) == optimal_cost``. On these tie-heavy maps this
  set is 3–5× larger than the reference path.
* `on_solver_path` — the one optimal path our solver happens to return, kept
  only as a tie-breaking-sensitive control.

Conventions, all deliberate and all logged in `docs/decisions.md`:

* **Outside the grid counts as obstacle.** Movement is confined to the grid, so
  a cell at the border is as blocked as one next to a wall. Every obstacle
  feature pads with obstacles before computing.
* **Obstacle cells get features too**, computed where meaningful and 0/NaN
  where not, so arrays stay aligned with the cost map. Step 5 selects rows with
  the `is_free` / `reachable` masks rather than relying on ragged arrays.
* **Unreachable free cells** (these maps have many — the goal sits in the
  largest component only) get ``np.inf`` for geodesic distances, which the
  masks exclude.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy import ndimage

from . import config
from .exact_solver import distance_field, solve_dijkstra
from .grid import RC, local_topology_class

#: Continuous features, in a fixed order — the column order of the design
#: matrix in step 5. `cost_map` (the regression target) and the mask/class
#: arrays are stored alongside but are not features.
FEATURE_NAMES: Tuple[str, ...] = (
    # goal geometry
    "geodesic_to_goal",
    "geodesic_from_start",
    "euclidean_to_goal",
    "chebyshev_to_goal",
    # obstacle geometry
    "dist_to_obstacle_chebyshev",
    "dist_to_obstacle_euclidean",
    "obstacle_density_3x3",
    "obstacle_density_5x5",
    "corridor_width_min4",
    "corridor_span_min",
    "free_neighbours",
    # path indicators
    "on_reference_path",
    "on_any_optimal_path",
    "on_solver_path",
)

#: Arrays stored next to the features: the target, the masks and the two
#: categorical partitions.
AUX_NAMES: Tuple[str, ...] = (
    "cost_map",
    "is_free",
    "reachable",
    "is_start",
    "is_goal",
    "topology_class",
    "wall_zone",
)

#: `topology_class` codes. PLAN.md asks for "dead end / corridor / junction /
#: open" from the free-neighbour count. These maps make the partition
#: degenerate (~68 % of free cells are OPEN, ~3 % are anything constrained;
#: `docs/verified.md` § 11), which is why `wall_zone` exists as well.
TOPOLOGY_OBSTACLE: int = -1
TOPOLOGY_DEAD_END: int = 0   # <= 1 free neighbours
TOPOLOGY_CORRIDOR: int = 1   # 2-3 free neighbours
TOPOLOGY_JUNCTION: int = 2   # 4-6 free neighbours
TOPOLOGY_OPEN: int = 3       # 7-8 free neighbours

TOPOLOGY_LABELS: Dict[int, str] = {
    TOPOLOGY_OBSTACLE: "obstacle",
    TOPOLOGY_DEAD_END: "dead_end",
    TOPOLOGY_CORRIDOR: "corridor",
    TOPOLOGY_JUNCTION: "junction",
    TOPOLOGY_OPEN: "open",
}

#: `wall_zone` codes: the partition that is actually informative on these maps,
#: built from the Chebyshev distance to the nearest obstacle.
WALL_ZONE_OBSTACLE: int = 0   # the wall itself
WALL_ZONE_ADJACENT: int = 1   # touching a wall (distance 1)
WALL_ZONE_NEAR: int = 2       # distance 2
WALL_ZONE_INTERIOR: int = 3   # distance >= 3, open space

WALL_ZONE_LABELS: Dict[int, str] = {
    WALL_ZONE_OBSTACLE: "obstacle",
    WALL_ZONE_ADJACENT: "wall_adjacent",
    WALL_ZONE_NEAR: "near_wall",
    WALL_ZONE_INTERIOR: "interior",
}


# --------------------------------------------------------------------------- #
# Goal geometry
# --------------------------------------------------------------------------- #

def euclidean_to_goal(shape: Tuple[int, int], goal_rc: RC) -> np.ndarray:
    """Straight-line distance from every cell to the goal.

    Args:
        shape: Grid shape ``(H, W)``.
        goal_rc: Goal cell.

    Returns:
        np.ndarray: Float64 ``(H, W)`` array, obstacles included (the quantity
        is purely positional).
    """
    rows = np.arange(shape[0], dtype=np.float64)[:, None]
    cols = np.arange(shape[1], dtype=np.float64)[None, :]
    return np.sqrt((rows - goal_rc[0]) ** 2 + (cols - goal_rc[1]) ** 2)


def chebyshev_to_goal(shape: Tuple[int, int], goal_rc: RC) -> np.ndarray:
    """Chebyshev distance to the goal — the upstream heuristic, less its tie-break.

    The search adds ``chebyshev + 0.001 * euclidean`` to ``h``
    (`differentiable_astar.py:26-52`), so this feature plus
    `euclidean_to_goal` span the heuristic exactly. A cost map that is just a
    rescaled heuristic is therefore detectable in step 5 as a two-feature fit.

    Args:
        shape: Grid shape ``(H, W)``.
        goal_rc: Goal cell.

    Returns:
        np.ndarray: Float64 ``(H, W)`` array.
    """
    rows = np.arange(shape[0], dtype=np.float64)[:, None]
    cols = np.arange(shape[1], dtype=np.float64)[None, :]
    dr = np.abs(rows - goal_rc[0])
    dc = np.abs(cols - goal_rc[1])
    return np.maximum(dr, dc)


# --------------------------------------------------------------------------- #
# Obstacle geometry
# --------------------------------------------------------------------------- #

def _pad_free(map_design: np.ndarray) -> np.ndarray:
    """Pad the free mask with a one-cell ring of obstacles.

    Args:
        map_design: ``(H, W)`` binary map, ``1`` = free.

    Returns:
        np.ndarray: ``(H+2, W+2)`` uint8 free mask whose border is 0, encoding
        the convention that outside the grid is blocked.
    """
    free = (map_design == config.FREE).astype(np.uint8)
    return np.pad(free, 1, mode="constant", constant_values=0)


def dist_to_obstacle(map_design: np.ndarray, metric: str = "chebyshev") -> np.ndarray:
    """Distance from each free cell to the nearest obstacle (or the grid edge).

    Args:
        map_design: ``(H, W)`` binary map, ``1`` = free.
        metric: ``"chebyshev"`` (the metric the 8-connected search moves in) or
            ``"euclidean"``.

    Returns:
        np.ndarray: Float64 ``(H, W)``. Obstacle cells get 0. A free cell
        touching a wall gets 1.

    Raises:
        ValueError: If `metric` is not recognised.
    """
    padded = _pad_free(map_design)
    if metric == "chebyshev":
        transformed = ndimage.distance_transform_cdt(padded, metric="chessboard")
    elif metric == "euclidean":
        transformed = ndimage.distance_transform_edt(padded)
    else:
        raise ValueError(f"unknown metric {metric!r}; expected chebyshev or euclidean")
    return np.asarray(transformed, dtype=np.float64)[1:-1, 1:-1]


def obstacle_density(map_design: np.ndarray, window: int) -> np.ndarray:
    """Fraction of obstacle cells in a square window centred on each cell.

    Args:
        map_design: ``(H, W)`` binary map, ``1`` = free.
        window: Odd window side (3 and 5 are the ones PLAN.md asks for).

    Returns:
        np.ndarray: Float64 ``(H, W)`` in ``[0, 1]``. Cells outside the grid
        count as obstacles, so border cells have a non-zero floor.

    Raises:
        ValueError: If `window` is not an odd positive integer.
    """
    if window < 1 or window % 2 == 0:
        raise ValueError(f"window must be odd and positive, got {window}")
    radius = window // 2
    free = (map_design == config.FREE).astype(np.float64)
    padded = np.pad(free, radius, mode="constant", constant_values=0.0)
    kernel = np.ones((window, window), dtype=np.float64)
    free_count = ndimage.correlate(padded, kernel, mode="constant", cval=0.0)
    free_count = free_count[radius:-radius, radius:-radius] if radius else free_count
    return 1.0 - free_count / float(window * window)


def _directional_distances(map_design: np.ndarray) -> Dict[str, np.ndarray]:
    """Steps to the nearest obstacle in each of the four orthogonal directions.

    A cell whose immediate neighbour in a direction is blocked (by an obstacle
    or by the grid edge) gets 1 for that direction.

    Args:
        map_design: ``(H, W)`` binary map, ``1`` = free.

    Returns:
        dict[str, np.ndarray]: Float64 ``(H, W)`` arrays keyed ``"up"``,
        ``"down"``, ``"left"``, ``"right"``. Obstacle cells get 0.
    """
    free = map_design == config.FREE
    height, width = map_design.shape
    out = {key: np.zeros(map_design.shape, dtype=np.float64) for key in ("up", "down", "left", "right")}

    for c in range(width):
        for r in range(height):
            if free[r, c]:
                out["left"][r, c] = 1.0 + (out["left"][r, c - 1] if c > 0 and free[r, c - 1] else 0.0)
    for c in range(width - 1, -1, -1):
        for r in range(height):
            if free[r, c]:
                out["right"][r, c] = 1.0 + (
                    out["right"][r, c + 1] if c + 1 < width and free[r, c + 1] else 0.0
                )
    for r in range(height):
        for c in range(width):
            if free[r, c]:
                out["up"][r, c] = 1.0 + (out["up"][r - 1, c] if r > 0 and free[r - 1, c] else 0.0)
    for r in range(height - 1, -1, -1):
        for c in range(width):
            if free[r, c]:
                out["down"][r, c] = 1.0 + (
                    out["down"][r + 1, c] if r + 1 < height and free[r + 1, c] else 0.0
                )
    return out


def corridor_width(map_design: np.ndarray) -> np.ndarray:
    """Corridor-width proxy: min over the four orthogonal directions.

    Exactly the definition the project settled on as its example of a choice to log:
    "corridor width defined as min over 4 orthogonal directions of distance to
    obstacle". A cell in the middle of a wide room scores high; a cell wedged
    against a wall scores 1 whatever the room's size.

    Args:
        map_design: ``(H, W)`` binary map, ``1`` = free.

    Returns:
        np.ndarray: Float64 ``(H, W)``; obstacle cells get 0.

    Note:
        This is *not* a width: it is a clearance. `corridor_span` is the width
        proxy proper, and both are stored because they disagree in exactly the
        places that matter (a cell against the wall of a wide room).
    """
    directions = _directional_distances(map_design)
    stacked = np.stack([directions[key] for key in ("up", "down", "left", "right")])
    width = stacked.min(axis=0)
    width[map_design != config.FREE] = 0.0
    return width


def corridor_span(map_design: np.ndarray) -> np.ndarray:
    """Width of the free span through each cell, minimised over the two axes.

    ``min(left + right - 1, up + down - 1)``: the number of free cells on the
    horizontal and vertical lines through the cell, whichever is narrower. A
    one-cell gap in a wall scores 1; the middle of a ten-wide room scores 10.

    Args:
        map_design: ``(H, W)`` binary map, ``1`` = free.

    Returns:
        np.ndarray: Float64 ``(H, W)``; obstacle cells get 0.
    """
    directions = _directional_distances(map_design)
    horizontal = directions["left"] + directions["right"] - 1.0
    vertical = directions["up"] + directions["down"] - 1.0
    span = np.minimum(horizontal, vertical)
    span[map_design != config.FREE] = 0.0
    return span


# --------------------------------------------------------------------------- #
# Path indicators
# --------------------------------------------------------------------------- #

def on_any_optimal_path(
    map_design: np.ndarray,
    start_rc: RC,
    goal_rc: RC,
    optimal_cost: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cells lying on *some* optimal start→goal path, plus the two distance fields.

    A cell is on an optimal path iff ``d(start, v) + d(v, goal) ==
    optimal_cost``. On these maps the set is 3–5× larger than any single
    optimal path, which is the quantitative face of the tie-heaviness PLAN.md
    § 5 worries about — and the reason step 5 must distinguish "marks the
    reference path" from "marks any shortest route".

    Args:
        map_design: ``(H, W)`` binary map, ``1`` = free.
        start_rc: Start cell.
        goal_rc: Goal cell.
        optimal_cost: Optional precomputed optimum; recomputed from the fields
            when omitted.

    Returns:
        tuple: ``(indicator, dist_from_start, dist_to_goal)`` — a float64
        0/1 ``(H, W)`` indicator and the two distance fields (``np.inf`` where
        unreachable).
    """
    dist_to_goal = distance_field(map_design, goal_rc)
    dist_from_start = distance_field(map_design, start_rc)
    if optimal_cost is None:
        optimal_cost = float(dist_to_goal[start_rc])
    total = dist_from_start + dist_to_goal
    with np.errstate(invalid="ignore"):
        indicator = (np.isfinite(total) & np.isclose(total, optimal_cost)).astype(np.float64)
    indicator[map_design != config.FREE] = 0.0
    return indicator, dist_from_start, dist_to_goal


def on_solver_path(map_design: np.ndarray, start_rc: RC, goal_rc: RC) -> np.ndarray:
    """Indicator of the single optimal path our solver returns.

    Args:
        map_design: ``(H, W)`` binary map, ``1`` = free.
        start_rc: Start cell.
        goal_rc: Goal cell.

    Returns:
        np.ndarray: Float64 ``(H, W)`` indicator, all zeros if unsolvable.

    Note:
        Which optimal path comes back depends on tie-breaking, so this feature
        is a control rather than a quantity to interpret: if it fits the cost
        map much better than `on_any_optimal_path`, something is wrong.
    """
    solution = solve_dijkstra(map_design, start_rc, goal_rc)
    out = np.zeros(map_design.shape, dtype=np.float64)
    for rc in solution.path:
        out[rc] = 1.0
    return out


# --------------------------------------------------------------------------- #
# Partitions
# --------------------------------------------------------------------------- #

def topology_class(map_design: np.ndarray) -> np.ndarray:
    """PLAN.md's dead-end / corridor / junction / open partition.

    Args:
        map_design: ``(H, W)`` binary map, ``1`` = free.

    Returns:
        np.ndarray: Int8 ``(H, W)`` of `TOPOLOGY_*` codes.

    Note:
        Degenerate on these maps (~68 % OPEN). Use `wall_zone` for the
        residual-by-topology table of step 5 and keep this one for
        comparability with PLAN.md as written.
    """
    neighbours = local_topology_class(map_design)
    out = np.full(map_design.shape, TOPOLOGY_OBSTACLE, dtype=np.int8)
    free = neighbours >= 0
    out[free & (neighbours <= 1)] = TOPOLOGY_DEAD_END
    out[free & (neighbours >= 2) & (neighbours <= 3)] = TOPOLOGY_CORRIDOR
    out[free & (neighbours >= 4) & (neighbours <= 6)] = TOPOLOGY_JUNCTION
    out[free & (neighbours >= 7)] = TOPOLOGY_OPEN
    return out


def wall_zone(map_design: np.ndarray) -> np.ndarray:
    """Partition by Chebyshev distance to the nearest obstacle.

    The partition that carries signal on thin-wall maps: wall-adjacent cells
    are where the planner's choices are constrained, and interior cells are
    where ties proliferate.

    Args:
        map_design: ``(H, W)`` binary map, ``1`` = free.

    Returns:
        np.ndarray: Int8 ``(H, W)`` of `WALL_ZONE_*` codes.
    """
    distance = dist_to_obstacle(map_design, metric="chebyshev")
    out = np.full(map_design.shape, WALL_ZONE_OBSTACLE, dtype=np.int8)
    out[distance == 1] = WALL_ZONE_ADJACENT
    out[distance == 2] = WALL_ZONE_NEAR
    out[distance >= 3] = WALL_ZONE_INTERIOR
    return out


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #

def compute_instance_features(
    instance,
    cost_map: Optional[np.ndarray] = None,
    reference_path: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Compute every feature and auxiliary array for one instance.

    Args:
        instance: A `src.data.MazeInstance`.
        cost_map: Optional ``(H, W)`` encoder output, stored as the regression
            target. Zeros are stored when omitted (useful for geometry-only
            use, e.g. choosing perturbation targets).
        reference_path: Optional ``(H, W)`` dataset trajectory from
            `src.data.MazeSplit.reference_trajectory`. Zeros when omitted,
            which disables the path-prediction test for that instance.

    Returns:
        dict[str, np.ndarray]: Float32 arrays for `FEATURE_NAMES`, plus the
        `AUX_NAMES` arrays (int8 for the two class arrays, float32 otherwise).
        Geodesic distances keep ``np.inf`` for unreachable cells; `reachable`
        marks where every geodesic feature is finite.

    Note:
        Costs ~5 ms per 32×32 instance, dominated by the two Dijkstra passes.
    """
    map_design = instance.map_design
    shape = map_design.shape

    any_path, dist_from_start, dist_to_goal = on_any_optimal_path(
        map_design, instance.start_rc, instance.goal_rc, instance.optimal_cost
    )
    free = (map_design == config.FREE).astype(np.float32)
    is_start = np.zeros(shape, dtype=np.float32)
    is_goal = np.zeros(shape, dtype=np.float32)
    is_start[instance.start_rc] = 1.0
    is_goal[instance.goal_rc] = 1.0

    features: Dict[str, np.ndarray] = {
        "geodesic_to_goal": dist_to_goal,
        "geodesic_from_start": dist_from_start,
        "euclidean_to_goal": euclidean_to_goal(shape, instance.goal_rc),
        "chebyshev_to_goal": chebyshev_to_goal(shape, instance.goal_rc),
        "dist_to_obstacle_chebyshev": dist_to_obstacle(map_design, "chebyshev"),
        "dist_to_obstacle_euclidean": dist_to_obstacle(map_design, "euclidean"),
        "obstacle_density_3x3": obstacle_density(map_design, 3),
        "obstacle_density_5x5": obstacle_density(map_design, 5),
        "corridor_width_min4": corridor_width(map_design),
        "corridor_span_min": corridor_span(map_design),
        "free_neighbours": np.maximum(local_topology_class(map_design), 0).astype(np.float64),
        "on_reference_path": (
            np.zeros(shape, dtype=np.float64) if reference_path is None else reference_path
        ),
        "on_any_optimal_path": any_path,
        "on_solver_path": on_solver_path(map_design, instance.start_rc, instance.goal_rc),
    }
    out = {name: np.asarray(features[name], dtype=np.float32) for name in FEATURE_NAMES}
    out["cost_map"] = np.asarray(
        np.zeros(shape) if cost_map is None else cost_map, dtype=np.float32
    )
    out["is_free"] = free
    out["reachable"] = (np.isfinite(dist_to_goal) & np.isfinite(dist_from_start)).astype(np.float32)
    out["is_start"] = is_start
    out["is_goal"] = is_goal
    out["topology_class"] = topology_class(map_design)
    out["wall_zone"] = wall_zone(map_design)
    return out
