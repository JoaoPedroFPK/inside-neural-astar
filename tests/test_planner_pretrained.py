"""The pretrained planner works, and works on *our* fixed-start instances.

This is the end-to-end sanity gate for the environment: if the checkpoint
loads into the architecture, classical A* reproduces the dataset's optimum
exactly, and Neural A* returns legal paths while expanding fewer nodes, then
the stack is wired correctly and E0's numbers mean something.

Kept to a 10-instance subset so the suite stays fast; the full-split E0
numbers are produced by `scripts/e0_reproduce.py`.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.grid import validate_path_map
from src.metrics import n_expansions, p_exp_reduction, p_opt_binary, path_cost
from src.planner import encode_cost_map

SUBSET = range(10)


@pytest.fixture(scope="module")
def batch(test_split):
    """A 10-instance batch with our fixed starts, plus the instances themselves."""
    instances = [test_split.instance(i, seed=0) for i in SUBSET]
    tensors = [inst.to_tensors() for inst in instances]
    maps = torch.cat([t[0] for t in tensors])
    starts = torch.cat([t[1] for t in tensors])
    goals = torch.cat([t[2] for t in tensors])
    return instances, maps, starts, goals


def test_checkpoint_loads_into_architecture(pretrained_planner) -> None:
    """The encoder is the 4-block CNN the checkpoint was trained with."""
    n_params = sum(p.numel() for p in pretrained_planner.encoder.parameters())
    assert n_params > 0
    assert not pretrained_planner.training, "planner must be in eval mode"


def test_cost_map_is_a_sigmoid_field(pretrained_planner, batch) -> None:
    """Cost maps are per-cell values in (0, 1), one channel, map-sized.

    The encoder ends in a sigmoid (`encoder.py:33-35`), so this also pins down
    the scale on which "cost of a path" is computed.
    """
    _, maps, starts, goals = batch
    costs = encode_cost_map(pretrained_planner, maps, starts, goals)
    assert costs.shape == maps.shape
    assert torch.all(costs > 0) and torch.all(costs < 1)
    # Not a constant field: it must carry spatial structure to be worth studying.
    assert costs.std().item() > 1e-3


def test_classical_astar_reproduces_dataset_optimum(vanilla_planner, batch) -> None:
    """Classical A* path cost equals the dataset's stored optimal distance.

    The strongest single check available without our own solver: it ties the
    search module, the unit-cost model, our fixed starts and the dataset's
    ground truth together in one equality.
    """
    instances, maps, starts, goals = batch
    out = vanilla_planner(maps, starts, goals)
    costs = path_cost(out.paths)
    expected = np.array([inst.optimal_cost for inst in instances])
    assert np.array_equal(costs, expected), (
        f"classical A* costs {costs} != dataset optima {expected}"
    )


def test_neural_astar_paths_are_valid_and_not_better_than_optimal(
    pretrained_planner, batch
) -> None:
    """Every Neural A* path is a legal chain whose cost is >= the optimum."""
    instances, maps, starts, goals = batch
    out = pretrained_planner(maps, starts, goals)
    costs = path_cost(out.paths)
    for i, inst in enumerate(instances):
        ok, reason = validate_path_map(
            out.paths[i, 0].numpy(), inst.map_design, inst.start_rc, inst.goal_rc
        )
        assert ok, f"instance {inst.instance_id}: {reason}"
        assert costs[i] >= inst.optimal_cost, (
            f"instance {inst.instance_id}: path cost {costs[i]} below the "
            f"optimum {inst.optimal_cost} - the cost model is wrong"
        )


def test_neural_astar_is_mostly_optimal_and_more_efficient(
    pretrained_planner, vanilla_planner, batch
) -> None:
    """The pretrained model behaves as advertised: near-optimal, fewer expansions.

    Thresholds are deliberately loose (this is a 10-instance smoke test, not
    the E0 measurement): most paths optimal, and strictly fewer nodes expanded
    on average than classical A*.
    """
    _, maps, starts, goals = batch
    neural = pretrained_planner(maps, starts, goals)
    vanilla = vanilla_planner(maps, starts, goals)

    p_opt = p_opt_binary(vanilla.paths, neural.paths).mean()
    p_exp = p_exp_reduction(vanilla.histories, neural.histories).mean()
    assert p_opt >= 0.5, f"only {p_opt:.0%} of paths optimal on the subset"
    assert p_exp > 0.0, "Neural A* expanded no fewer nodes than classical A*"
    assert n_expansions(neural.histories).sum() < n_expansions(vanilla.histories).sum()


def test_planner_is_deterministic(pretrained_planner, batch) -> None:
    """Two identical calls give identical results (no hidden randomness)."""
    _, maps, starts, goals = batch
    first = pretrained_planner(maps, starts, goals)
    second = pretrained_planner(maps, starts, goals)
    assert torch.equal(first.paths, second.paths)
    assert torch.equal(first.histories, second.histories)


def test_classical_astar_matches_dataset_optimum_on_whole_split(
    vanilla_planner, test_split
) -> None:
    """The same equality as above, over all 100 test instances.

    Run on the full split because it is the one assertion that validates the
    cost model end to end (dataset ground truth <-> unit-cost moves <-> the
    search module <-> our fixed starts). Costs ~1 s.
    """
    instances = test_split.instances(seed=0)
    tensors = [inst.to_tensors() for inst in instances]
    maps = torch.cat([t[0] for t in tensors])
    starts = torch.cat([t[1] for t in tensors])
    goals = torch.cat([t[2] for t in tensors])

    with torch.no_grad():
        out = vanilla_planner(maps, starts, goals)

    costs = path_cost(out.paths)
    expected = np.array([inst.optimal_cost for inst in instances])
    mismatched = np.flatnonzero(costs != expected)
    assert mismatched.size == 0, (
        f"{mismatched.size}/{len(instances)} instances disagree, e.g. "
        f"instance {mismatched[0]}: A* cost {costs[mismatched[0]]} vs dataset "
        f"optimum {expected[mismatched[0]]}"
    )


def test_cost_map_change_from_one_flip_is_confined_to_receptive_field(
    pretrained_planner, test_split
) -> None:
    """A k=1 flip changes the cost map only within radius 5 of the flipped cell.

    The encoder is five 3x3 convolutions with padding 1 (`encoder.py:62-70`:
    `channels = [input_dim] + CHANNELS[:depth] + [1]` gives *five* conv blocks
    for depth 4), so its receptive field has radius 5. In eval mode BatchNorm
    uses running statistics and mixes nothing spatially, so the bound is exact,
    not approximate - and it bounds the encoder route of the mediation
    analysis: a behaviour change outside this window cannot have come from the
    encoder's reading of that region.
    """
    from src.grid import flip_cells
    from src.planner import encode_cost_map

    radius = sum(
        (m.kernel_size[0] - 1) // 2
        for m in pretrained_planner.encoder.model
        if isinstance(m, torch.nn.Conv2d)
    )
    assert radius == 5, f"receptive-field radius changed to {radius}; update the docs"

    rng = np.random.default_rng(0)
    rows, cols = np.ogrid[: config_map_size(), : config_map_size()]
    for i in range(3):
        inst = test_split.instance(i, seed=0)
        maps, starts, goals = inst.to_tensors()
        base = encode_cost_map(pretrained_planner, maps, starts, goals)[0, 0].numpy()
        for _ in range(3):
            rc = (int(rng.integers(base.shape[0])), int(rng.integers(base.shape[1])))
            if rc in (inst.start_rc, inst.goal_rc):
                continue
            altered = torch.from_numpy(flip_cells(inst.map_design, [rc])).reshape_as(maps)
            delta = np.abs(
                encode_cost_map(pretrained_planner, altered, starts, goals)[0, 0].numpy() - base
            )
            outside = (np.abs(rows - rc[0]) > radius) | (np.abs(cols - rc[1]) > radius)
            assert delta[outside].max() == 0.0, (
                f"instance {i}, flip {rc}: cost map changed outside radius {radius}"
            )
            assert delta.max() > 0.0, f"instance {i}, flip {rc}: cost map did not change at all"


def config_map_size() -> int:
    """Grid side length, read from the config (helper for the test above)."""
    from src import config

    return config.MAP_SIZE
