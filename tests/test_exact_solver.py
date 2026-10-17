"""PLAN.md step 2: the exact solver.

The done-criterion test is
`test_distance_field_matches_dataset_on_all_1000_maps`: on every unaltered map
of every split, the solver's geodesic distances must equal the distances stored
in the dataset, exactly.

The other high-value tests check the solver against the *upstream search
module* rather than against itself:
`test_astar_reproduces_upstream_vanilla_expansions` and
`test_solver_reproduces_neural_astar_search` assert that our numpy search is a
drop-in replica of `differentiable_astar`, including the two details that are
easy to get wrong (cost charged at the parent, cost map added to ``h``).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src import config
from src.data import load_split
from src.exact_solver import (
    chebyshev_heuristic,
    distance_field,
    path_cost_on,
    solve_astar,
    solve_dijkstra,
)
from src.grid import validate_path_map
from src.metrics import n_expansions, path_cost
from src.planner import encode_cost_map

SUBSET = range(25)


@pytest.fixture(scope="module")
def instances(test_split):
    """A 25-instance subset of the test split with fixed starts."""
    return [test_split.instance(i, seed=0) for i in SUBSET]


@pytest.fixture(scope="module")
def tensor_batch(instances):
    """The same subset as planner-ready tensors."""
    tensors = [inst.to_tensors() for inst in instances]
    return (
        torch.cat([t[0] for t in tensors]),
        torch.cat([t[1] for t in tensors]),
        torch.cat([t[2] for t in tensors]),
    )


# --------------------------------------------------------------------------- #
# The done-criterion
# --------------------------------------------------------------------------- #

def test_distance_field_matches_dataset_on_all_1000_maps() -> None:
    """Solver distances == stored distances, on all 1000 maps, every free cell.

    PLAN.md step 2's done-criterion. Covers train (800), valid (100) and test
    (100), and checks the two notions of "unreachable" agree as well.
    """
    checked_maps = 0
    checked_cells = 0
    for split_name in ("train", "valid", "test"):
        split = load_split(split_name)
        for i in range(len(split)):
            map_design = split.map_designs[i]
            stored = split.opt_dists[i, 0]
            field = distance_field(map_design, split.goal_rc(i))

            free = map_design == config.FREE
            both = free & np.isfinite(field) & np.isfinite(stored)
            assert np.array_equal(field[both], stored[both]), (
                f"{split_name}[{i}]: solver distances disagree with the dataset"
            )
            assert np.array_equal(np.isfinite(field) & free, np.isfinite(stored) & free), (
                f"{split_name}[{i}]: disagreement on which cells are reachable"
            )
            assert not np.isfinite(field[~free]).any(), (
                f"{split_name}[{i}]: solver assigned a finite distance to an obstacle"
            )
            checked_maps += 1
            checked_cells += int(both.sum())

    assert checked_maps == sum(config.SPLIT_SIZES.values()) == 1000
    assert checked_cells > 300_000  # sanity: the check really ran over the grids


# --------------------------------------------------------------------------- #
# Internal consistency
# --------------------------------------------------------------------------- #

def test_dijkstra_matches_dataset_optimum_at_fixed_starts(test_split) -> None:
    """Point-to-point Dijkstra agrees with the stored distance field."""
    for i in range(len(test_split)):
        inst = test_split.instance(i, seed=0)
        solution = solve_dijkstra(inst.map_design, inst.start_rc, inst.goal_rc)
        assert solution.solved
        assert solution.cost == inst.optimal_cost, f"instance {i}"


def test_astar_cost_equals_dijkstra_cost(instances) -> None:
    """Under unit costs, upstream's heuristic does not cost optimality.

    The Euclidean tie-break term overestimates by at most ~0.044 on a 32x32
    grid, below the minimum integer cost gap of 1 - checked rather than assumed.
    """
    for inst in instances:
        a = solve_astar(inst.map_design, inst.start_rc, inst.goal_rc)
        d = solve_dijkstra(inst.map_design, inst.start_rc, inst.goal_rc)
        assert a.cost == d.cost == inst.optimal_cost, f"instance {inst.instance_id}"


def test_returned_paths_are_valid_chains(instances) -> None:
    """Paths are simple 8-connected chains whose length matches the cost."""
    for inst in instances:
        solution = solve_astar(inst.map_design, inst.start_rc, inst.goal_rc)
        assert solution.path[0] == inst.start_rc
        assert solution.path[-1] == inst.goal_rc
        assert solution.n_moves == solution.cost
        raster = np.zeros_like(inst.map_design)
        for rc in solution.path:
            raster[rc] = 1
        ok, reason = validate_path_map(raster, inst.map_design, inst.start_rc, inst.goal_rc)
        assert ok, f"instance {inst.instance_id}: {reason}"
        assert path_cost_on(solution.path) == solution.cost


def test_distance_field_agrees_with_point_to_point_weighted(instances) -> None:
    """With a non-uniform cost map, the field still equals per-start Dijkstra."""
    rng = np.random.default_rng(0)
    inst = instances[0]
    cost_map = rng.uniform(0.1, 1.0, size=inst.map_design.shape)
    field = distance_field(inst.map_design, inst.goal_rc, cost_map=cost_map)

    free_cells = np.argwhere((inst.map_design == 1) & np.isfinite(field))
    for rc in free_cells[:: max(1, len(free_cells) // 20)]:
        rc = (int(rc[0]), int(rc[1]))
        if rc == inst.goal_rc:
            continue
        point = solve_dijkstra(inst.map_design, rc, inst.goal_rc, cost_map=cost_map)
        assert np.isclose(point.cost, field[rc]), f"cell {rc}"
        assert np.isclose(path_cost_on(point.path, cost_map), point.cost)


def test_weighted_astar_is_never_cheaper_than_dijkstra(instances, tensor_batch, pretrained_planner) -> None:
    """With a learned cost map, A* may be suboptimal but never "better".

    The cost map is folded into ``h`` upstream, so the search is weighted and
    its path can cost more than the optimum - which is one of the mechanisms
    this project studies. It must never cost *less*.
    """
    maps, starts, goals = tensor_batch
    cost_maps = encode_cost_map(pretrained_planner, maps, starts, goals).numpy()[:, 0]
    for k, inst in enumerate(instances):
        cm = cost_maps[k].astype(np.float64)
        a = solve_astar(inst.map_design, inst.start_rc, inst.goal_rc, cost_map=cm)
        d = solve_dijkstra(inst.map_design, inst.start_rc, inst.goal_rc, cost_map=cm)
        assert a.cost >= d.cost - 1e-9, f"instance {inst.instance_id}"


# --------------------------------------------------------------------------- #
# Agreement with the upstream search module
# --------------------------------------------------------------------------- #

def test_astar_reproduces_upstream_vanilla_expansions(
    instances, tensor_batch, vanilla_planner
) -> None:
    """Our classical A* closes exactly the nodes upstream's VanillaAstar does.

    Equality of *expansion counts*, not just costs, is what makes `Exp` a
    well-defined ratio: the denominator is the same number whichever
    implementation produces it.
    """
    maps, starts, goals = tensor_batch
    with torch.no_grad():
        out = vanilla_planner(maps, starts, goals)
    upstream_exp = n_expansions(out.histories)
    upstream_cost = path_cost(out.paths)

    for k, inst in enumerate(instances):
        ours = solve_astar(inst.map_design, inst.start_rc, inst.goal_rc)
        assert ours.expansions == upstream_exp[k], (
            f"instance {inst.instance_id}: {ours.expansions} expansions vs "
            f"upstream {upstream_exp[k]}"
        )
        assert ours.cost == upstream_cost[k]


def test_solver_reproduces_neural_astar_search(
    instances, tensor_batch, pretrained_planner
) -> None:
    """Fed Neural A*'s cost map, the solver reproduces its path and expansions.

    This is the end-to-end confirmation of the cost-application semantics in
    `docs/verified.md` section 4: cost charged at the *parent* of each expansion
    **and** added to ``h``. Get either detail wrong and these counts diverge.
    """
    maps, starts, goals = tensor_batch
    with torch.no_grad():
        out = pretrained_planner(maps, starts, goals)
    module_exp = n_expansions(out.histories)
    module_moves = path_cost(out.paths)
    cost_maps = encode_cost_map(pretrained_planner, maps, starts, goals).numpy()[:, 0]

    for k, inst in enumerate(instances):
        ours = solve_astar(
            inst.map_design, inst.start_rc, inst.goal_rc, cost_map=cost_maps[k].astype(np.float64)
        )
        assert ours.expansions == module_exp[k], (
            f"instance {inst.instance_id}: {ours.expansions} expansions vs module {module_exp[k]}"
        )
        assert ours.n_moves == module_moves[k], (
            f"instance {inst.instance_id}: {ours.n_moves} moves vs module {module_moves[k]}"
        )


def test_heuristic_matches_upstream_implementation(pretrained_planner) -> None:
    """`chebyshev_heuristic` equals upstream's `get_heuristic`, elementwise."""
    from neural_astar.planner.differentiable_astar import get_heuristic

    goal_rc = (7, 19)
    goal = torch.zeros(1, config.MAP_SIZE, config.MAP_SIZE)
    goal[0, goal_rc[0], goal_rc[1]] = 1.0
    upstream = get_heuristic(goal)[0].numpy()
    ours = chebyshev_heuristic((config.MAP_SIZE, config.MAP_SIZE), goal_rc)
    assert np.allclose(ours, upstream, atol=1e-6)


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #

def test_unreachable_goal_reports_failure() -> None:
    """A sealed goal yields `solved=False`, infinite cost and an empty path."""
    m = np.zeros((5, 5), dtype=np.float32)
    m[2, :] = 1.0
    m[2, 2] = 0.0  # seal the corridor
    solution = solve_dijkstra(m, (2, 0), (2, 4))
    assert not solution.solved
    assert solution.cost == float("inf")
    assert solution.path == []
    assert np.isinf(distance_field(m, (2, 4))[2, 0])


def test_invalid_inputs_are_rejected() -> None:
    """Obstacle endpoints, out-of-bounds cells and bad cost maps raise."""
    m = np.ones((4, 4), dtype=np.float32)
    m[1, 1] = 0.0
    with pytest.raises(ValueError, match="start"):
        solve_astar(m, (1, 1), (3, 3))
    with pytest.raises(ValueError, match="goal"):
        solve_astar(m, (0, 0), (1, 1))
    with pytest.raises(ValueError, match="outside"):
        solve_astar(m, (0, 0), (9, 9))
    with pytest.raises(ValueError, match="cost map shape"):
        solve_astar(m, (0, 0), (3, 3), cost_map=np.ones((2, 2)))
    with pytest.raises(ValueError, match="negative costs"):
        solve_astar(m, (0, 0), (3, 3), cost_map=-np.ones((4, 4)))
    with pytest.raises(ValueError, match="no cost"):
        path_cost_on([(0, 0)])


def test_expansion_count_includes_start_and_goal() -> None:
    """Expansions are counted as upstream counts them: start and goal included."""
    m = np.ones((1, 3), dtype=np.float32)
    solution = solve_astar(m, (0, 0), (0, 2))
    assert solution.path == [(0, 0), (0, 1), (0, 2)]
    assert solution.expansions == 3


def test_solver_agrees_with_upstream_on_altered_maps(test_split, vanilla_planner) -> None:
    """The agreement also holds on flipped maps — the regime steps 3/6 use.

    Every earlier check runs on unaltered maps, where the dataset provides an
    independent reference. The perturbation studies have no such reference, so
    the solver has to be trusted there: this test flips one admissible cell per
    instance and re-checks cost and expansion agreement against the upstream
    module on the altered map.
    """
    from src.grid import flip_cells, is_reachable

    rng = np.random.default_rng(0)
    n_checked = 0
    for i in range(10):
        inst = test_split.instance(i, seed=0)
        candidates = [
            (int(r), int(c))
            for r, c in np.argwhere(np.ones_like(inst.map_design))
            if (int(r), int(c)) not in (inst.start_rc, inst.goal_rc)
        ]
        rng.shuffle(candidates)
        altered = None
        for rc in candidates:
            candidate = flip_cells(inst.map_design, [rc])
            if is_reachable(candidate, inst.start_rc, inst.goal_rc):
                altered = candidate
                break
        assert altered is not None, f"instance {i}: no admissible flip found"

        maps = torch.from_numpy(altered).reshape(1, 1, *altered.shape)
        starts = torch.zeros_like(maps)
        goals = torch.zeros_like(maps)
        starts[0, 0][inst.start_rc] = 1.0
        goals[0, 0][inst.goal_rc] = 1.0
        with torch.no_grad():
            out = vanilla_planner(maps, starts, goals)

        ours = solve_astar(altered, inst.start_rc, inst.goal_rc)
        dijkstra = solve_dijkstra(altered, inst.start_rc, inst.goal_rc)
        assert ours.expansions == n_expansions(out.histories)[0], f"instance {i}: expansions"
        assert ours.cost == path_cost(out.paths)[0], f"instance {i}: cost"
        assert ours.cost == dijkstra.cost, f"instance {i}: A* vs Dijkstra on altered map"
        n_checked += 1
    assert n_checked == 10
