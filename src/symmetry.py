"""Rotation symmetry of the planner: the same problem, turned by 90, 180, 270 degrees.

A grid path-finding problem has the symmetries of the square. Rotating the map,
the start and the goal together gives a problem with the same optimal cost, and
an exact planner with a rotation-invariant heuristic returns a route of that
cost on every copy. Upstream's heuristic (Chebyshev distance plus a small
Euclidean tie-break, `src.exact_solver.chebyshev_heuristic`) and its unit move
costs are both invariant under quarter turns, so the *search* has no preferred
orientation. Only two things in Neural A* can break the symmetry:

* the **encoder**, a stack of learned 3x3 convolutions that nothing forces to be
  rotation-equivariant: ``encode(rot(map))`` need not equal ``rot(encode(map))``;
* **tie-breaking** among nodes with equal score, which follows row-major cell
  order and therefore does depend on orientation. This affects *which* of
  several equally short routes is returned, not its length; classical A* is
  subject to it too, which is why it is the control here.

This module turns an instance, runs the planner and the exact references on all
copies in one batch, rotates every output back into the original frame, and
measures the gaps. The numpy part (rotations, the classical control) needs no
torch; the planner part imports it lazily.

Conventions: ``k`` is the number of counter-clockwise quarter turns, exactly as
in ``np.rot90(array, k)``. "Back in the original frame" means ``np.rot90(x, -k)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .data import MazeInstance
from .exact_solver import solve_astar, solve_dijkstra
from .grid import RC

#: The four quarter turns studied.
ROTATIONS: Tuple[int, ...] = (0, 1, 2, 3)
#: Degrees of each quarter turn, for labels.
DEGREES: Dict[int, int] = {0: 0, 1: 90, 2: 180, 3: 270}
#: Path costs closer than this are "the same cost".
COST_TOL: float = 1e-6


# --------------------------------------------------------------------------- geometry
def rotate_rc(rc: RC, k: int, shape: Tuple[int, int]) -> RC:
    """Where cell `rc` of an ``(H, W)`` array ends up after ``np.rot90(·, k)``.

    One counter-clockwise quarter turn sends ``(r, c)`` to ``(W - 1 - c, r)``
    and the shape to ``(W, H)``; `k` turns apply that `k mod 4` times.

    Args:
        rc: Cell ``(row, col)`` in the unrotated array.
        k: Number of counter-clockwise quarter turns (any integer).
        shape: Shape ``(H, W)`` of the unrotated array.

    Returns:
        tuple[int, int]: The cell in the rotated array.
    """
    r, c = int(rc[0]), int(rc[1])
    h, w = int(shape[0]), int(shape[1])
    for _ in range(k % 4):
        r, c = w - 1 - c, r
        h, w = w, h
    return (r, c)


def rotate_cells(cells: Sequence[RC], k: int, shape: Tuple[int, int]) -> List[RC]:
    """`rotate_rc` applied to every cell of a list, order kept."""
    return [rotate_rc(rc, k, shape) for rc in cells]


def unrotate(array: np.ndarray, k: int) -> np.ndarray:
    """Bring an array computed on the ``k``-rotated copy back to the original frame."""
    return np.ascontiguousarray(np.rot90(array, -k))


def rotate_instance(instance: MazeInstance, k: int) -> MazeInstance:
    """The same planning problem turned by ``k`` counter-clockwise quarter turns.

    Map, start, goal and the dataset's distance field are all rotated, so the
    rotated instance is a valid problem with the same optimal cost.
    `instance_id`, `split` and `start_seed` are kept: they identify the
    *problem*, and the rotation is recorded by the caller.

    Args:
        instance: The original instance.
        k: Counter-clockwise quarter turns.

    Returns:
        MazeInstance: A new instance; the input is not modified.
    """
    shape = instance.map_design.shape
    return replace(
        instance,
        map_design=np.ascontiguousarray(np.rot90(instance.map_design, k)),
        start_rc=rotate_rc(instance.start_rc, k, shape),
        goal_rc=rotate_rc(instance.goal_rc, k, shape),
        opt_dist=np.ascontiguousarray(np.rot90(instance.opt_dist, k)),
    )


def path_to_map(path: Sequence[RC], shape: Tuple[int, int]) -> np.ndarray:
    """A list of cells as a binary ``(H, W)`` map."""
    out = np.zeros(shape, dtype=np.float32)
    for rc in path:
        out[tuple(rc)] = 1.0
    return out


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two binary maps (1.0 for two empty maps)."""
    a, b = a > 0, b > 0
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


# --------------------------------------------------------------------------- classical control
@dataclass
class ClassicalRotation:
    """Exact references on one rotated copy, mapped back to the original frame.

    Attributes:
        k: Quarter turns.
        optimal_cost: Certified optimum (Dijkstra) on the rotated copy.
        astar_cost: Cost of classical A*'s route (upstream heuristic, unit costs).
        astar_expansions: Classical A*'s expansion count.
        astar_path_map: Classical A*'s route, rotated back, as a binary map.
    """

    k: int
    optimal_cost: float
    astar_cost: float
    astar_expansions: float
    astar_path_map: np.ndarray


def classical_rotations(instance: MazeInstance, ks: Sequence[int] = ROTATIONS) -> List[ClassicalRotation]:
    """Run the exact solver and classical A* on every rotated copy (numpy only).

    This is the control: whatever classical A* does under rotation is what the
    problem and the tie-breaking rule allow, not a defect of a learned model.

    Args:
        instance: The original instance.
        ks: Quarter turns to evaluate.

    Returns:
        list[ClassicalRotation]: One entry per `k`, in the order given.
    """
    out = []
    for k in ks:
        rot = rotate_instance(instance, k)
        optimum = solve_dijkstra(rot.map_design, rot.start_rc, rot.goal_rc)
        astar = solve_astar(rot.map_design, rot.start_rc, rot.goal_rc)
        out.append(
            ClassicalRotation(
                k=int(k),
                optimal_cost=float(optimum.cost),
                astar_cost=float(astar.cost),
                astar_expansions=float(astar.expansions),
                astar_path_map=unrotate(path_to_map(astar.path, rot.map_design.shape), k),
            )
        )
    return out


# --------------------------------------------------------------------------- the planner
@dataclass
class PlannerRotation:
    """Neural A* on one rotated copy; every map is rotated back to the original frame.

    Attributes:
        k: Quarter turns.
        cost_map: ``(H, W)`` encoder output.
        path_map: ``(H, W)`` binary route.
        histories: ``(H, W)`` binary map of expanded cells.
        path_cost: Number of moves on the route (``-1`` if no route).
        expansions: Number of expanded cells.
        optimal_cost: Certified optimum on the copy (equal for every `k`).
        classical: The classical control on the same copy.
    """

    k: int
    cost_map: np.ndarray
    path_map: np.ndarray
    histories: np.ndarray
    path_cost: float
    expansions: float
    optimal_cost: float
    classical: ClassicalRotation

    @property
    def opt_ratio(self) -> float:
        """Route length over the optimum (1.0 = shortest)."""
        return self.path_cost / self.optimal_cost

    @property
    def exp_ratio(self) -> float:
        """Expansions over classical A*'s on the same copy."""
        return self.expansions / self.classical.astar_expansions


def evaluate_rotations(
    planner,
    instance: MazeInstance,
    ks: Sequence[int] = ROTATIONS,
    cost_map_fn=None,
) -> List[PlannerRotation]:
    """Run Neural A* on every rotated copy of one instance, in one batch.

    Args:
        planner: A `NeuralAstar` in eval mode.
        instance: The original instance.
        ks: Quarter turns to evaluate.
        cost_map_fn: Optional ``(maps, starts, goals) -> cost maps`` replacing
            the encoder (used for the symmetrised planner,
            `symmetrised_cost_maps`). Defaults to the planner's own encoder.

    Returns:
        list[PlannerRotation]: One entry per `k`, in the order given, with
        every map in the original frame so entries compare cell by cell.
    """
    import torch

    from .metrics import n_expansions, path_cost
    from .planner import encode_cost_map, run_astar_module

    rotated = [rotate_instance(instance, k) for k in ks]
    tensors = [r.to_tensors() for r in rotated]
    maps = torch.cat([t[0] for t in tensors])
    starts = torch.cat([t[1] for t in tensors])
    goals = torch.cat([t[2] for t in tensors])
    with torch.no_grad():
        costs = (cost_map_fn or (lambda m, s, g: encode_cost_map(planner, m, s, g)))(maps, starts, goals)
        out = run_astar_module(planner, costs, starts, goals, maps)
    moves = path_cost(out.paths)
    expanded = n_expansions(out.histories)
    classical = classical_rotations(instance, ks)

    results = []
    for i, k in enumerate(ks):
        results.append(
            PlannerRotation(
                k=int(k),
                cost_map=unrotate(costs[i, 0].cpu().numpy(), k),
                path_map=unrotate(out.paths[i, 0].cpu().numpy(), k),
                histories=unrotate(out.histories[i, 0].cpu().numpy(), k),
                path_cost=float(moves[i]),
                expansions=float(expanded[i]),
                optimal_cost=classical[i].optimal_cost,
                classical=classical[i],
            )
        )
    return results


def symmetrised_cost_maps(planner, ks: Sequence[int] = ROTATIONS):
    """A cost-map function that averages the encoder over all quarter turns.

    For each input map, the encoder is run on its four rotations, the outputs
    are rotated back and averaged, and the average is rotated into the input's
    frame. The result is exactly rotation-equivariant by construction (test-
    time augmentation over the rotation group). Planning on it isolates what is
    left once the encoder's asymmetry is removed: only tie-breaking.

    Args:
        planner: A `NeuralAstar` in eval mode.
        ks: The group elements averaged over (all four quarter turns by default).

    Returns:
        Callable: ``(maps, starts, goals) -> cost maps``, each ``(B, 1, H, W)``.
    """
    import torch

    from .planner import encode_cost_map

    def fn(maps, starts, goals):
        acc = torch.zeros_like(maps)
        for k in ks:
            rot = lambda x: torch.rot90(x, k, dims=(2, 3))  # noqa: E731
            enc = encode_cost_map(planner, rot(maps), rot(starts), rot(goals))
            acc += torch.rot90(enc, -k, dims=(2, 3))
        return acc / len(ks)

    return fn


# --------------------------------------------------------------------------- summaries
@dataclass
class SymmetryReport:
    """Rotation gaps of one instance, relative to the unrotated copy.

    Attributes:
        instance_id: The instance.
        rows: One dict per rotation (see `summarize_rotations`).
        cost_spread: ``(H, W)`` standard deviation of the four rotated-back cost
            maps, per cell: where the encoder disagrees with itself.
        same_cost: The planner's route length is equal on every copy.
        same_route: The planner's route is the same cells on every copy.
        classical_same_route: Classical A*'s route is the same on every copy.
    """

    instance_id: int
    rows: List[Dict[str, float]] = field(default_factory=list)
    cost_spread: Optional[np.ndarray] = None
    same_cost: bool = True
    same_route: bool = True
    classical_same_route: bool = True


def summarize_rotations(results: Sequence[PlannerRotation], free: Optional[np.ndarray] = None,
                        instance_id: int = -1) -> SymmetryReport:
    """Compare every rotated copy with the first one (normally ``k = 0``).

    Columns per rotation:

    * ``path_cost``, ``optimal_cost``, ``opt_ratio``, ``expansions``,
      ``exp_ratio``, ``classical_expansions`` — the project's usual metrics;
    * ``d_path_cost`` — route length minus the unrotated copy's;
    * ``route_iou`` / ``classical_route_iou`` — overlap of the rotated-back
      route with the unrotated one (1.0 = the same cells);
    * ``expanded_iou`` — the same for the expanded cells;
    * ``cost_mae``, ``cost_max``, ``cost_r`` — the encoder's equivariance gap
      on free cells: mean and largest absolute difference and Pearson r
      between the rotated-back cost map and the unrotated one.

    Args:
        results: Output of `evaluate_rotations`; the first entry is the
            reference.
        free: Optional ``(H, W)`` mask of cells to compare cost maps on; free
            cells of the map are the natural choice (walls are never read).
        instance_id: Recorded in the report.

    Returns:
        SymmetryReport: The rows and the per-instance verdicts.
    """
    ref = results[0]
    mask = np.ones_like(ref.cost_map, dtype=bool) if free is None else free > 0
    report = SymmetryReport(instance_id=instance_id)
    for res in results:
        a, b = res.cost_map[mask], ref.cost_map[mask]
        diff = np.abs(a - b)
        r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 1.0
        report.rows.append(
            {
                "instance_id": instance_id,
                "k": res.k,
                "degrees": DEGREES[res.k % 4],
                "path_cost": res.path_cost,
                "optimal_cost": res.optimal_cost,
                "opt_ratio": res.opt_ratio,
                "d_path_cost": res.path_cost - ref.path_cost,
                "expansions": res.expansions,
                "classical_expansions": res.classical.astar_expansions,
                "exp_ratio": res.exp_ratio,
                "route_iou": iou(res.path_map, ref.path_map),
                "classical_route_iou": iou(res.classical.astar_path_map, ref.classical.astar_path_map),
                "classical_cost": res.classical.astar_cost,
                "expanded_iou": iou(res.histories, ref.histories),
                "cost_mae": float(diff.mean()),
                "cost_max": float(diff.max()),
                "cost_r": r,
            }
        )
    stack = np.stack([res.cost_map for res in results])
    report.cost_spread = np.where(mask, stack.std(axis=0), 0.0)
    report.same_cost = all(abs(row["d_path_cost"]) < COST_TOL for row in report.rows)
    report.same_route = all(row["route_iou"] > 1 - 1e-9 for row in report.rows)
    report.classical_same_route = all(row["classical_route_iou"] > 1 - 1e-9 for row in report.rows)
    return report


def rotation_study(planner, instances: Sequence[MazeInstance], ks: Sequence[int] = ROTATIONS,
                   cost_map_fn=None) -> Tuple[List[Dict[str, float]], List[SymmetryReport]]:
    """`evaluate_rotations` + `summarize_rotations` over many instances.

    Args:
        planner: A `NeuralAstar` in eval mode.
        instances: Original instances.
        ks: Quarter turns.
        cost_map_fn: Forwarded to `evaluate_rotations`.

    Returns:
        tuple[list[dict], list[SymmetryReport]]: All rows (ready for
        ``pandas.DataFrame``) and one report per instance.
    """
    rows, reports = [], []
    for inst in instances:
        res = evaluate_rotations(planner, inst, ks, cost_map_fn=cost_map_fn)
        rep = summarize_rotations(res, free=inst.map_design, instance_id=inst.instance_id)
        rows.extend(rep.rows)
        reports.append(rep)
    return rows, reports
