"""Grid geometry on the 8-connected ("moore") maze grids — pure numpy.

Everything in this module uses the **unit-cost** move model: all eight moves
(four orthogonal and four diagonal) cost 1. That is not an arbitrary choice; it
is what the whole upstream stack does, and matching it is what makes our
"optimal" the same optimum the planner is searching for:

* `neural_astar/planner/differentiable_astar.py:234` propagates
  ``g2 = g[parent] + cost_map[parent]`` to all eight neighbours, with no
  diagonal factor;
* `neural_astar/planner/pq_astar.py:138-144` likewise adds
  ``g_ratio * cost[neighbour]`` regardless of direction;
* the dataset's stored distances come from
  `planning-datasets/planning_datasets_utils/dijkstra.py:153`
  (``curr_to_n = val + 1``).

The consequence, and the reason the upstream heuristic is Chebyshev, is that
the shortest-path metric on an empty grid is the Chebyshev distance.
See `docs/verified.md` (checks 3 and 4) and `docs/decisions.md` (2026-10-17).

This module is deliberately torch-free so that it can be used from feature
extraction, from the exact solver and from tests without loading torch.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable, Iterator, Sequence, Tuple

import numpy as np

#: The eight moves of the Moore neighbourhood as ``(dr, dc)`` offsets.
#: Order is deterministic (row-major) so that anything iterating over
#: neighbours is reproducible, including tie-breaking in our own solvers.
MOVES_8: Tuple[Tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)

#: Cost of a single move under the upstream model (see module docstring).
MOVE_COST: float = 1.0

RC = Tuple[int, int]


def in_bounds(rc: RC, shape: Tuple[int, int]) -> bool:
    """Return whether a ``(row, col)`` pair lies inside a grid of `shape`.

    Args:
        rc: Candidate ``(row, col)`` index.
        shape: Grid shape ``(H, W)``.

    Returns:
        bool: True if ``0 <= row < H`` and ``0 <= col < W``.
    """
    r, c = rc
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors8(rc: RC, shape: Tuple[int, int]) -> Iterator[RC]:
    """Yield the in-bounds Moore neighbours of a cell, in `MOVES_8` order.

    Obstacles are *not* filtered here; callers that need traversability should
    test the map themselves (this keeps the function usable for neighbourhood
    statistics in the E1 feature library, where obstacle neighbours matter).

    Args:
        rc: Cell ``(row, col)``.
        shape: Grid shape ``(H, W)``.

    Yields:
        tuple[int, int]: Neighbouring ``(row, col)`` indices inside the grid.
    """
    r, c = rc
    for dr, dc in MOVES_8:
        cand = (r + dr, c + dc)
        if in_bounds(cand, shape):
            yield cand


def bfs_distances(map_design: np.ndarray, source_rc: RC) -> np.ndarray:
    """Unit-cost 8-connected geodesic distances from `source_rc` to every cell.

    Because every move costs exactly 1 (see module docstring), a breadth-first
    search is already a shortest-path algorithm here — no priority queue is
    needed. This is the reference implementation the dataset's stored
    distances are checked against in `tests/test_data.py`.

    Args:
        map_design: Binary map of shape ``(H, W)``; ``1`` = free,
            ``0`` = obstacle (`src.config.FREE` / `src.config.OBSTACLE`).
        source_rc: Source cell ``(row, col)``. Must be a free cell.

    Returns:
        np.ndarray: Float array of shape ``(H, W)``. Entry ``(r, c)`` is the
        number of moves on a shortest free path from `source_rc` to ``(r, c)``,
        or ``np.inf`` for obstacles and for free cells in a different
        connected component.

    Raises:
        ValueError: If `map_design` is not 2-D, or `source_rc` is out of bounds
            or sits on an obstacle.

    Example:
        >>> m = np.ones((3, 3), dtype=np.float32)
        >>> bfs_distances(m, (1, 1))[0, 0]   # diagonal move costs 1
        1.0
    """
    if map_design.ndim != 2:
        raise ValueError(f"expected a 2-D map, got shape {map_design.shape}")
    if not in_bounds(source_rc, map_design.shape):
        raise ValueError(f"source {source_rc} outside map of shape {map_design.shape}")
    if map_design[source_rc] != 1:
        raise ValueError(f"source {source_rc} is an obstacle")

    dist = np.full(map_design.shape, np.inf, dtype=np.float64)
    dist[source_rc] = 0.0
    queue: deque[RC] = deque([source_rc])
    while queue:
        cur = queue.popleft()
        for nb in neighbors8(cur, map_design.shape):
            if map_design[nb] == 1 and not np.isfinite(dist[nb]):
                dist[nb] = dist[cur] + MOVE_COST
                queue.append(nb)
    return dist


def is_reachable(map_design: np.ndarray, start_rc: RC, goal_rc: RC) -> bool:
    """Return whether `goal_rc` can be reached from `start_rc` on free cells.

    This is the predicate behind reachability constraint: every
    altered map must keep the goal reachable from the (fixed) start, and flips
    that violate it are rejected rather than counted as failures.

    Args:
        map_design: Binary map, ``1`` = free.
        start_rc: Start cell ``(row, col)``.
        goal_rc: Goal cell ``(row, col)``.

    Returns:
        bool: True if a free 8-connected path exists. False if either endpoint
        became an obstacle, or no path exists.
    """
    if not (in_bounds(start_rc, map_design.shape) and in_bounds(goal_rc, map_design.shape)):
        return False
    if map_design[start_rc] != 1 or map_design[goal_rc] != 1:
        return False
    if start_rc == goal_rc:
        return True
    # Early-exit BFS: cheaper than computing the full distance field.
    seen = np.zeros(map_design.shape, dtype=bool)
    seen[start_rc] = True
    queue: deque[RC] = deque([start_rc])
    while queue:
        cur = queue.popleft()
        for nb in neighbors8(cur, map_design.shape):
            if map_design[nb] == 1 and not seen[nb]:
                if nb == goal_rc:
                    return True
                seen[nb] = True
                queue.append(nb)
    return False


def flip_cells(map_design: np.ndarray, cells: Iterable[RC]) -> np.ndarray:
    """Return a copy of `map_design` with the given cells' polarity inverted.

    A "flip" is the atomic map alteration of this project (PLAN.md steps 3, 6
    and 9): a free cell becomes an obstacle or an obstacle becomes free. The
    input is never modified, so cached original maps stay clean.

    Args:
        map_design: Binary map of shape ``(H, W)``, values in ``{0, 1}``.
        cells: Iterable of ``(row, col)`` cells to flip. Repeated cells flip
            repeatedly (flipping the same cell twice restores it), which keeps
            the operation a pure function of the cell multiset.

    Returns:
        np.ndarray: New array of the same dtype and shape, with
        ``out[cell] = 1 - map_design[cell]`` for each listed cell.

    Raises:
        ValueError: If a cell is out of bounds.
    """
    out = map_design.copy()
    for rc in cells:
        if not in_bounds(rc, out.shape):
            raise ValueError(f"cell {rc} outside map of shape {out.shape}")
        out[rc] = 1 - out[rc]
    return out


def free_cells(map_design: np.ndarray) -> np.ndarray:
    """Return the ``(N, 2)`` array of ``(row, col)`` indices of free cells.

    Args:
        map_design: Binary map, ``1`` = free.

    Returns:
        np.ndarray: Integer array of shape ``(N, 2)`` in row-major order.
    """
    return np.argwhere(map_design == 1)


def local_topology_class(map_design: np.ndarray) -> np.ndarray:
    """Classify every free cell by its number of free Moore neighbours.

    Provided here because both the E1 feature library (PLAN.md step 4) and the
    residual-by-topology table (step 5) need exactly this quantity, and it is
    pure grid geometry.

    Args:
        map_design: Binary map, ``1`` = free.

    Returns:
        np.ndarray: Integer array of shape ``(H, W)``. Obstacles get ``-1``;
        free cells get their count of free neighbours in ``0..8``. The
        coarse classes used in the write-up are derived from this count in
        `src.features` (step 4), not here, so the raw count stays available.
    """
    free = (map_design == 1).astype(np.int16)
    padded = np.pad(free, 1, mode="constant", constant_values=0)
    counts = np.zeros_like(free, dtype=np.int16)
    for dr, dc in MOVES_8:
        counts += padded[1 + dr : 1 + dr + free.shape[0], 1 + dc : 1 + dc + free.shape[1]]
    out = np.where(free == 1, counts, -1).astype(np.int16)
    return out


def path_cells_from_map(path_map: np.ndarray) -> Sequence[RC]:
    """Return the set of cells marked on a binary path map, row-major.

    The upstream A* modules return paths as binary maps rather than ordered
    sequences; this is the unordered view used for overlap statistics (e.g.
    the path-indicator feature of PLAN.md step 4).

    Args:
        path_map: Binary array of shape ``(H, W)``.

    Returns:
        list[tuple[int, int]]: Cells with value 1, in row-major order.
    """
    return [tuple(rc) for rc in np.argwhere(path_map > 0)]


def validate_path_map(
    path_map: np.ndarray,
    map_design: np.ndarray,
    start_rc: RC,
    goal_rc: RC,
) -> Tuple[bool, str]:
    """Check that a planner's binary path map is a legal simple path.

    The upstream planners return paths as binary maps rather than ordered
    sequences, so "is this a path at all?" has to be re-derived. A legal output
    here is a *simple chain*: every marked cell is free, the start and the goal
    are marked, the marked cells are 8-connected, and walking the chain from
    start to goal visits all of them (no side branches, no detached blobs).

    Used as the correctness gate on every swap condition in E3 — an
    off-distribution ``(cost map, obstacle mask)`` pair must still produce a
    path that respects the mask it was given, or the comparison is meaningless.

    Args:
        path_map: ``(H, W)`` binary path map (``AstarOutput.paths[i, 0]``).
        map_design: ``(H, W)`` binary map whose obstacles the path must respect
            — i.e. the *mask that was passed to the search*, which under a swap
            is not necessarily the map the cost came from.
        start_rc: Expected start cell.
        goal_rc: Expected goal cell.

    Returns:
        tuple[bool, str]: ``(True, "ok")`` if the path is legal, otherwise
        ``(False, reason)`` with a short human-readable reason.

    Example:
        >>> m = np.ones((3, 3), dtype=np.float32)
        >>> p = np.zeros((3, 3)); p[0, 0] = p[1, 1] = p[2, 2] = 1
        >>> validate_path_map(p, m, (0, 0), (2, 2))
        (True, 'ok')
    """
    cells = np.argwhere(path_map > 0)
    if len(cells) == 0:
        return False, "empty path map (search returned no path)"
    if path_map[start_rc] == 0:
        return False, f"start {start_rc} not on the path"
    if path_map[goal_rc] == 0:
        return False, f"goal {goal_rc} not on the path"
    blocked = [tuple(rc) for rc in cells if map_design[tuple(rc)] != 1]
    if blocked:
        return False, f"path crosses {len(blocked)} obstacle cell(s), e.g. {blocked[0]}"

    sub = np.zeros_like(map_design, dtype=np.float32)
    sub[path_map > 0] = 1.0
    dist = bfs_distances(sub, start_rc)
    if not np.isfinite(dist[goal_rc]):
        return False, "path cells are not connected from start to goal"
    if dist[goal_rc] != len(cells) - 1:
        return (
            False,
            f"path is not a simple chain: {len(cells)} cells but "
            f"{int(dist[goal_rc])} moves from start to goal",
        )
    return True, "ok"
