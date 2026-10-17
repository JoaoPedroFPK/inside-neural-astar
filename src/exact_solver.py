"""Exact search on the same grid, cost model and heuristic as upstream.

Two solvers, for two different jobs — the distinction matters and is easy to
blur:

`solve_dijkstra` / `distance_field`
    **Ground truth for ``Opt``.** No heuristic, so the returned cost is the
    true optimum whatever the cost map looks like. This is what an altered
    map's optimum is measured with (project rule: an alteration is an error only
    if Neural A\\* deviates from the *altered map's* optimum).

`solve_astar`
    **The classical-A\\* control for ``Exp``.** Replicates upstream's search
    rule — Chebyshev + ``0.001 × Euclidean`` heuristic, ``f = g_ratio · g +
    (1 - g_ratio) · (h + cost)``, no reopening of closed nodes — so its
    expansion count is comparable with the planner's.

Both follow the cost model verified in `docs/verified.md` §§ 4-5:

* all eight moves cost the same (no √2 for diagonals);
* the cost of a path ``v0 … vn`` is ``Σ cost[v_i]`` for ``i < n`` — charged at
  the **parent** of each expansion, so the goal's own cost is never paid;
* with ``cost_map = None`` every free cell costs 1, which is exactly what
  `VanillaAstar` does by passing the obstacle map as the cost map.

On the tie-breaking term: upstream's heuristic is ``Chebyshev + 0.001 ×
Euclidean``, which *overestimates* the true distance by up to
``0.001 × 31 × √2 ≈ 0.044`` on a 32×32 grid. Under unit costs every path cost
is an integer, so an f-ordering error below 1.0 cannot make a more expensive
path win: A\\* here is optimal in practice, and
`tests/test_exact_solver.py::test_astar_cost_equals_dijkstra_cost` checks it on
every instance rather than relying on the argument. With a *learned* cost map
the heuristic is not admissible at all (costs are added to ``h``), which is one
mechanism by which Neural A\\* can return a suboptimal path — so for ground
truth, use Dijkstra.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .grid import MOVES_8, RC

#: Upstream's tie-breaking weight on the Euclidean term
#: (`differentiable_astar.py:26`, `get_heuristic(..., tb_factor=0.001)`).
TB_FACTOR: float = 0.001

#: Upstream's default weight of ``g`` in ``f`` (`differentiable_astar.py:129`).
G_RATIO: float = 0.5


@dataclass(frozen=True)
class ExactSolution:
    """Result of an exact search.

    Attributes:
        cost: Total path cost under the unit-cost model — the number of moves
            when `cost_map` is None, otherwise ``Σ cost[parent]`` along the
            path. ``np.inf`` if the goal is unreachable.
        path: Cells from start to goal inclusive, in order. Empty if the goal
            is unreachable.
        expansions: Number of nodes closed by the search, counted the way
            upstream counts them: start and goal included, each node once
            (`differentiable_astar.py:222`, where the selected node is added to
            ``histories`` before the termination check).
        solved: Whether a path was found.
    """

    cost: float
    path: List[RC]
    expansions: int
    solved: bool

    @property
    def n_moves(self) -> int:
        """Number of moves on the path (``len(path) - 1``), or -1 if unsolved."""
        return len(self.path) - 1 if self.solved else -1


def chebyshev_heuristic(
    shape: Tuple[int, int], goal_rc: RC, tb_factor: float = TB_FACTOR
) -> np.ndarray:
    """Upstream's heuristic field: Chebyshev distance plus a Euclidean tie-break.

    Reimplements `neural_astar.planner.differentiable_astar.get_heuristic`
    (lines 26-52) in numpy, for one goal:

    ``h = (|dr| + |dc| - min(|dr|, |dc|)) + tb_factor * sqrt(dr² + dc²)``

    Args:
        shape: Grid shape ``(H, W)``.
        goal_rc: Goal cell ``(row, col)``.
        tb_factor: Weight of the Euclidean tie-breaking term. The default is
            upstream's.

    Returns:
        np.ndarray: Float64 array of shape ``(H, W)``.
    """
    rows = np.arange(shape[0], dtype=np.float64)[:, None]
    cols = np.arange(shape[1], dtype=np.float64)[None, :]
    dr = np.abs(rows - goal_rc[0])
    dc = np.abs(cols - goal_rc[1])
    chebyshev = dr + dc - np.minimum(dr, dc)
    euclidean = np.sqrt(dr**2 + dc**2)
    return chebyshev + tb_factor * euclidean


def _prepare(
    map_design: np.ndarray, cost_map: Optional[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Validate inputs and return flattened traversability and cost arrays.

    Args:
        map_design: ``(H, W)`` binary map, ``1`` = free.
        cost_map: Optional ``(H, W)`` per-cell cost. ``None`` means unit cost
            on every free cell.

    Returns:
        tuple: ``(free_flat, cost_flat, H, W)`` where `free_flat` is a boolean
        array and `cost_flat` is float64, both flattened row-major.

    Raises:
        ValueError: If shapes disagree, the map is not 2-D, or any cost on a
            free cell is negative (which would break Dijkstra's optimality).
    """
    if map_design.ndim != 2:
        raise ValueError(f"expected a 2-D map, got shape {map_design.shape}")
    free = map_design == 1
    if cost_map is None:
        cost = np.ones(map_design.shape, dtype=np.float64)
    else:
        cost_map = np.asarray(cost_map, dtype=np.float64)
        if cost_map.shape != map_design.shape:
            raise ValueError(
                f"cost map shape {cost_map.shape} != map shape {map_design.shape}"
            )
        if np.any(cost_map[free] < 0):
            raise ValueError("negative costs on free cells are not supported")
        cost = cost_map
    return free.ravel(), cost.ravel(), map_design.shape[0], map_design.shape[1]


def _neighbor_offsets(width: int) -> List[Tuple[int, int, int]]:
    """Flat-index offsets of the eight moves, with their row/col deltas.

    Args:
        width: Grid width ``W``.

    Returns:
        list[tuple[int, int, int]]: ``(dr, dc, flat_offset)`` per move, in
        `src.grid.MOVES_8` order.
    """
    return [(dr, dc, dr * width + dc) for dr, dc in MOVES_8]


def _reconstruct(parent: np.ndarray, start_idx: int, goal_idx: int, width: int) -> List[RC]:
    """Walk the parent array back from the goal and return the path.

    Args:
        parent: Flat array of parent indices, ``-1`` where unset.
        start_idx: Flat index of the start cell.
        goal_idx: Flat index of the goal cell.
        width: Grid width ``W``.

    Returns:
        list[tuple[int, int]]: Cells from start to goal inclusive.
    """
    path_flat = [goal_idx]
    while path_flat[-1] != start_idx:
        nxt = int(parent[path_flat[-1]])
        if nxt < 0:  # pragma: no cover - only reachable on a corrupt parent array
            raise RuntimeError("parent chain broken before reaching the start")
        path_flat.append(nxt)
    path_flat.reverse()
    return [(idx // width, idx % width) for idx in path_flat]


def _search(
    map_design: np.ndarray,
    start_rc: RC,
    goal_rc: RC,
    cost_map: Optional[np.ndarray],
    heuristic: Optional[np.ndarray],
    g_ratio: float,
) -> ExactSolution:
    """Shared best-first search; `heuristic=None` makes it Dijkstra.

    Node selection minimises ``f = g_ratio * g + (1 - g_ratio) * (h + cost[v])``
    when a heuristic is given, and ``g`` alone otherwise. Ties are broken by the
    lowest flat (row-major) index, which is how upstream breaks them too
    (`torch.max` returns the first maximal index). Closed nodes are never
    reopened, matching `differentiable_astar.py:235`.

    Args:
        map_design: ``(H, W)`` binary map, ``1`` = free.
        start_rc: Start cell.
        goal_rc: Goal cell.
        cost_map: Optional per-cell cost; ``None`` = unit cost.
        heuristic: Optional precomputed ``(H, W)`` heuristic field.
        g_ratio: Weight of ``g`` in ``f``; ignored when `heuristic` is None.

    Returns:
        ExactSolution: Cost, path, expansion count and solved flag.

    Raises:
        ValueError: If an endpoint is out of bounds or sits on an obstacle.
    """
    free, cost, height, width = _prepare(map_design, cost_map)
    for name, rc in (("start", start_rc), ("goal", goal_rc)):
        if not (0 <= rc[0] < height and 0 <= rc[1] < width):
            raise ValueError(f"{name} {rc} outside map of shape {(height, width)}")
        if not free[rc[0] * width + rc[1]]:
            raise ValueError(f"{name} {rc} is an obstacle")

    start_idx = start_rc[0] * width + start_rc[1]
    goal_idx = goal_rc[0] * width + goal_rc[1]
    h_flat = (
        np.zeros(height * width, dtype=np.float64)
        if heuristic is None
        else heuristic.ravel()
    )
    use_h = heuristic is not None

    g = np.full(height * width, np.inf, dtype=np.float64)
    parent = np.full(height * width, -1, dtype=np.int64)
    closed = np.zeros(height * width, dtype=bool)
    g[start_idx] = 0.0

    def priority(idx: int, g_value: float) -> float:
        """f-value of a node, matching upstream's weighting."""
        if not use_h:
            return g_value
        return g_ratio * g_value + (1.0 - g_ratio) * (h_flat[idx] + cost[idx])

    offsets = _neighbor_offsets(width)
    heap: List[Tuple[float, int]] = [(priority(start_idx, 0.0), start_idx)]
    expansions = 0

    while heap:
        _, idx = heapq.heappop(heap)
        if closed[idx]:
            continue
        closed[idx] = True
        expansions += 1

        if idx == goal_idx:
            return ExactSolution(
                cost=float(g[idx]),
                path=_reconstruct(parent, start_idx, goal_idx, width),
                expansions=expansions,
                solved=True,
            )

        row, col = divmod(idx, width)
        step_cost = cost[idx]  # charged at the parent (docs/verified.md section 4)
        for dr, dc, offset in offsets:
            nr, nc = row + dr, col + dc
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            nb = idx + offset
            if not free[nb] or closed[nb]:
                continue
            tentative = g[idx] + step_cost
            if tentative < g[nb]:
                g[nb] = tentative
                parent[nb] = idx
                heapq.heappush(heap, (priority(nb, tentative), nb))

    return ExactSolution(cost=float("inf"), path=[], expansions=expansions, solved=False)


def solve_dijkstra(
    map_design: np.ndarray,
    start_rc: RC,
    goal_rc: RC,
    cost_map: Optional[np.ndarray] = None,
) -> ExactSolution:
    """Exact shortest path with no heuristic — the ground truth for ``Opt``.

    Use this (not `solve_astar`) whenever the number is going to be used as
    "the optimum", because it does not depend on a heuristic being admissible.

    Args:
        map_design: ``(H, W)`` binary map, ``1`` = free.
        start_rc: Start cell.
        goal_rc: Goal cell.
        cost_map: Optional per-cell cost charged at the parent of each
            expansion; ``None`` = unit cost on free cells.

    Returns:
        ExactSolution: With `cost` equal to the optimal path cost (the number
        of moves under unit costs), or ``np.inf`` if the goal is unreachable.

    Example:
        >>> m = np.ones((3, 3), dtype=np.float32)
        >>> solve_dijkstra(m, (0, 0), (2, 2)).cost   # one diagonal per move
        2.0
    """
    return _search(map_design, start_rc, goal_rc, cost_map, None, G_RATIO)


def solve_astar(
    map_design: np.ndarray,
    start_rc: RC,
    goal_rc: RC,
    cost_map: Optional[np.ndarray] = None,
    g_ratio: float = G_RATIO,
    tb_factor: float = TB_FACTOR,
) -> ExactSolution:
    """A\\* with upstream's heuristic — the classical control for ``Exp``.

    Args:
        map_design: ``(H, W)`` binary map, ``1`` = free.
        start_rc: Start cell.
        goal_rc: Goal cell.
        cost_map: Optional per-cell cost; ``None`` = unit cost, which makes
            this the exact analogue of `neural_astar.planner.VanillaAstar`.
        g_ratio: Weight of ``g`` in ``f``. Must match the planner being
            compared against (upstream's default is 0.5).
        tb_factor: Weight of the Euclidean tie-breaking term.

    Returns:
        ExactSolution: Cost, path and expansion count. The expansion count is
        the quantity ``Exp`` divides by.

    Note:
        The heuristic absorbs the cost map (``h + cost[v]``, as upstream does
        at `differentiable_astar.py:192`), so with a learned cost map this is a
        *weighted* search and its path need not be optimal. That is the
        intended behaviour — it is the same rule the planner follows.
    """
    heuristic = chebyshev_heuristic(map_design.shape, goal_rc, tb_factor)
    return _search(map_design, start_rc, goal_rc, cost_map, heuristic, g_ratio)


def distance_field(
    map_design: np.ndarray,
    goal_rc: RC,
    cost_map: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Optimal cost from **every** cell to the goal, in one backward Dijkstra.

    This is the workhorse for the perturbation sweeps: one call gives the
    optimum for every possible start on an altered map, so an exhaustive flip
    study needs one Dijkstra per map rather than one per (map, start) pair.
    It is also what the step-2 unit test compares against the dataset's stored
    distances.

    The search runs backwards from the goal. Under the parent-charged cost
    model, relaxing an edge ``u -> v`` backwards adds ``cost[v]`` (the forward
    parent), so for unit costs this is a plain BFS and reproduces the dataset's
    distances exactly.

    Args:
        map_design: ``(H, W)`` binary map, ``1`` = free.
        goal_rc: Goal cell.
        cost_map: Optional per-cell cost charged at the forward parent;
            ``None`` = unit cost.

    Returns:
        np.ndarray: Float64 ``(H, W)`` field of optimal costs to the goal, with
        ``np.inf`` on obstacles and on free cells with no path to the goal.

    Raises:
        ValueError: If the goal is out of bounds or on an obstacle.
    """
    free, cost, height, width = _prepare(map_design, cost_map)
    if not (0 <= goal_rc[0] < height and 0 <= goal_rc[1] < width):
        raise ValueError(f"goal {goal_rc} outside map of shape {(height, width)}")
    goal_idx = goal_rc[0] * width + goal_rc[1]
    if not free[goal_idx]:
        raise ValueError(f"goal {goal_rc} is an obstacle")

    dist = np.full(height * width, np.inf, dtype=np.float64)
    closed = np.zeros(height * width, dtype=bool)
    dist[goal_idx] = 0.0
    offsets = _neighbor_offsets(width)
    heap: List[Tuple[float, int]] = [(0.0, goal_idx)]

    while heap:
        d, idx = heapq.heappop(heap)
        if closed[idx]:
            continue
        closed[idx] = True
        row, col = divmod(idx, width)
        for dr, dc, offset in offsets:
            nr, nc = row + dr, col + dc
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            nb = idx + offset
            if not free[nb] or closed[nb]:
                continue
            # Backward relaxation: the move nb -> idx pays cost[nb].
            tentative = d + cost[nb]
            if tentative < dist[nb]:
                dist[nb] = tentative
                heapq.heappush(heap, (tentative, nb))

    return dist.reshape(height, width)


def path_cost_on(path: List[RC], cost_map: Optional[np.ndarray] = None) -> float:
    """Cost of an explicit path under the parent-charged cost model.

    Args:
        path: Cells from start to goal inclusive.
        cost_map: Optional ``(H, W)`` per-cell cost; ``None`` = unit cost, in
            which case the result is ``len(path) - 1``.

    Returns:
        float: ``Σ cost[v_i]`` over all cells except the last, or ``np.inf`` for
        an empty path.

    Raises:
        ValueError: If the path has a single cell and is therefore not a move.
    """
    if not path:
        return float("inf")
    if len(path) == 1:
        raise ValueError("a path of one cell has no cost; start == goal?")
    if cost_map is None:
        return float(len(path) - 1)
    return float(sum(cost_map[rc] for rc in path[:-1]))
