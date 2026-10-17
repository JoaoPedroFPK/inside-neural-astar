"""PLAN.md step 1: the swap design is structurally possible.

The 2x2 swap design of E3 assumes the differentiable A* module takes the cost
map and the obstacle map as *separate* inputs, and that the obstacle map acts
only as a mask. These tests exercise that assumption instead of trusting the
reading of the source:

* `differentiable_astar.py:150-156` - `forward(cost_maps, start_maps,
  goal_maps, obstacles_maps, ...)`;
* `differentiable_astar.py:229` - ``neighbor_nodes * obstacles_maps``, the
  only use of the obstacle map in the module.

The decisive test is `test_cross_instance_swap_returns_valid_path`: costs from
instance A, obstacles from instance B.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.grid import validate_path_map
from src.planner import encode_cost_map, run_astar_module


@pytest.fixture(scope="module")
def pair(test_split):
    """Two test instances (A and B) that share nothing but their grid size."""
    return test_split.instance(0, seed=0), test_split.instance(1, seed=0)


def test_module_signature_separates_costs_and_obstacles(pretrained_planner) -> None:
    """The search module's signature must expose costs and obstacles separately."""
    import inspect

    params = list(inspect.signature(pretrained_planner.astar.forward).parameters)
    assert params[:4] == ["cost_maps", "start_maps", "goal_maps", "obstacles_maps"], (
        f"unexpected signature {params}; the swap design assumes these four inputs"
    )


def test_forward_equals_encode_then_search(pretrained_planner, pair) -> None:
    """Calling the module by hand reproduces `NeuralAstar.forward` exactly.

    Guarantees that the swap harness's `C0O0` baseline is the same computation
    as a plain forward pass, so baseline numbers are comparable to E0's.
    """
    inst, _ = pair
    map_designs, start_maps, goal_maps = inst.to_tensors()

    reference = pretrained_planner(map_designs, start_maps, goal_maps)
    costs = encode_cost_map(pretrained_planner, map_designs, start_maps, goal_maps)
    manual = run_astar_module(
        pretrained_planner, costs, start_maps, goal_maps, map_designs
    )

    assert torch.equal(reference.paths, manual.paths)
    assert torch.equal(reference.histories, manual.histories)


def test_cross_instance_swap_returns_valid_path(pretrained_planner, pair) -> None:
    """Costs from instance A + obstacles from instance B still plan a valid path.

    This is the test the project asks for ("run the module with costs from
    instance A and obstacles from instance B; it must return a valid path").
    Start and goal come from B, since they must be free cells of B's map.
    """
    inst_a, inst_b = pair
    maps_a, starts_a, goals_a = inst_a.to_tensors()
    maps_b, starts_b, goals_b = inst_b.to_tensors()

    costs_a = encode_cost_map(pretrained_planner, maps_a, starts_a, goals_a)
    out = run_astar_module(pretrained_planner, costs_a, starts_b, goals_b, maps_b)

    path = out.paths[0, 0].numpy()
    ok, reason = validate_path_map(path, inst_b.map_design, inst_b.start_rc, inst_b.goal_rc)
    assert ok, f"cross-instance swap produced an invalid path: {reason}"


def test_obstacle_mask_alone_decides_traversability(pretrained_planner, pair) -> None:
    """The mask, not the cost map, decides where the search may go.

    With obstacles from B and costs from A, no expanded node and no path cell
    may fall on an obstacle of B, however cheap A's cost map says it is.
    """
    inst_a, inst_b = pair
    maps_a, starts_a, goals_a = inst_a.to_tensors()
    maps_b, starts_b, goals_b = inst_b.to_tensors()

    costs_a = encode_cost_map(pretrained_planner, maps_a, starts_a, goals_a)
    out = run_astar_module(pretrained_planner, costs_a, starts_b, goals_b, maps_b)

    obstacles_b = inst_b.map_design == 0
    expanded = out.histories[0, 0].numpy() > 0
    on_path = out.paths[0, 0].numpy() > 0
    # The start itself is always "expanded" and is free by construction.
    assert not np.any(expanded & obstacles_b), "search expanded an obstacle cell"
    assert not np.any(on_path & obstacles_b), "path crosses an obstacle cell"


def test_cost_map_alone_changes_the_search(pretrained_planner, pair) -> None:
    """Swapping only the cost map changes the search, so the route is live.

    If the module ignored the cost map (or folded obstacles into it), `C1O0`
    and `C0O0` would be identical and the mediation analysis would be vacuous.
    """
    inst_a, inst_b = pair
    maps_a, starts_a, goals_a = inst_a.to_tensors()
    maps_b, starts_b, goals_b = inst_b.to_tensors()

    own_costs = encode_cost_map(pretrained_planner, maps_b, starts_b, goals_b)
    foreign_costs = encode_cost_map(pretrained_planner, maps_a, starts_a, goals_a)

    own = run_astar_module(pretrained_planner, own_costs, starts_b, goals_b, maps_b)
    foreign = run_astar_module(pretrained_planner, foreign_costs, starts_b, goals_b, maps_b)

    assert not torch.equal(own_costs, foreign_costs), "encoder ignored its input map"
    assert not torch.equal(own.histories, foreign.histories), (
        "the cost map had no effect on the search; the encoder route cannot be "
        "measured this way"
    )


def test_uniform_cost_map_reproduces_classical_astar(
    pretrained_planner, vanilla_planner, pair
) -> None:
    """Feeding the obstacle map as the cost map is exactly classical A*.

    `VanillaAstar.forward` does precisely this
    (`neural_astar/planner/astar.py:93-94`), which is what makes "classical A*
    on the same map" a condition of the same harness rather than a separate
    implementation.
    """
    inst, _ = pair
    map_designs, start_maps, goal_maps = inst.to_tensors()

    reference = vanilla_planner(map_designs, start_maps, goal_maps)
    manual = run_astar_module(
        pretrained_planner, map_designs, start_maps, goal_maps, map_designs
    )

    assert torch.equal(reference.paths, manual.paths)
    assert torch.equal(reference.histories, manual.histories)


def test_all_four_swap_conditions_run(pretrained_planner, test_split) -> None:
    """A real k=1 flip runs through all four conditions and yields valid paths.

    A miniature of the E3 harness: flip one free cell of the instance's own
    map, keep the goal reachable, and run C0O0 / C0O1 / C1O0 / C1O1. Each
    condition must return a path that is legal **for the obstacle mask it was
    given**.
    """
    from src.grid import flip_cells, is_reachable

    inst = test_split.instance(0, seed=0)
    original = inst.map_design

    # First free, non-terminal cell whose removal keeps start->goal connected.
    flip_cell = None
    for rc in np.argwhere(original == 1):
        rc = (int(rc[0]), int(rc[1]))
        if rc in (inst.start_rc, inst.goal_rc):
            continue
        candidate = flip_cells(original, [rc])
        if is_reachable(candidate, inst.start_rc, inst.goal_rc):
            flip_cell = rc
            break
    assert flip_cell is not None, "no admissible k=1 flip found"
    altered = flip_cells(original, [flip_cell])

    maps0, starts, goals = inst.to_tensors()
    maps1 = torch.from_numpy(altered).reshape_as(maps0)

    costs0 = encode_cost_map(pretrained_planner, maps0, starts, goals)
    costs1 = encode_cost_map(pretrained_planner, maps1, starts, goals)

    conditions = {
        "C0O0": (costs0, maps0, original),
        "C0O1": (costs0, maps1, altered),
        "C1O0": (costs1, maps0, original),
        "C1O1": (costs1, maps1, altered),
    }
    for label, (costs, mask_tensor, mask_array) in conditions.items():
        out = run_astar_module(pretrained_planner, costs, starts, goals, mask_tensor)
        ok, reason = validate_path_map(
            out.paths[0, 0].numpy(), mask_array, inst.start_rc, inst.goal_rc
        )
        assert ok, f"{label}: {reason}"
