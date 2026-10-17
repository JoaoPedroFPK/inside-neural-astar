"""PLAN.md step 3: the flip sampler and the alteration evaluation harness."""

from __future__ import annotations

import numpy as np
import pytest

from src import config
from src.grid import flip_cells, is_reachable
from src.perturb import (
    AlterationRequest,
    candidate_cells,
    compute_baseline,
    enumerate_k1_flips,
    evaluate_alterations,
    sample_random_flips,
    summarize_by_direction,
    summarize_rows,
)


@pytest.fixture(scope="module")
def instance(test_split):
    """One test instance with a fixed start."""
    return test_split.instance(0, seed=0)


# --------------------------------------------------------------------------- #
# Candidate sets and sampling
# --------------------------------------------------------------------------- #

def test_candidate_cells_excludes_start_and_goal(instance) -> None:
    """Every cell but the start and the goal is a candidate."""
    cells = candidate_cells(instance.map_design, instance.start_rc, instance.goal_rc)
    assert len(cells) == config.MAP_SIZE**2 - 2
    assert instance.start_rc not in cells
    assert instance.goal_rc not in cells
    assert cells == sorted(cells), "candidates must be in row-major order"


def test_candidate_cells_free_only(instance) -> None:
    """`free_only` keeps obstacle-adding flips only."""
    cells = candidate_cells(
        instance.map_design, instance.start_rc, instance.goal_rc, free_only=True
    )
    assert all(instance.map_design[rc] == config.FREE for rc in cells)
    assert len(cells) < config.MAP_SIZE**2 - 2


def test_sample_random_flips_is_admissible_and_deterministic(instance) -> None:
    """Draws have the right size, keep the goal reachable, and repeat per seed."""
    for budget in (1, 2, 4, 8):
        first = sample_random_flips(instance, budget, np.random.default_rng([0, budget]))
        second = sample_random_flips(instance, budget, np.random.default_rng([0, budget]))
        assert first == second, "same seed must give the same draw"
        assert first is not None and len(set(first)) == budget
        assert instance.start_rc not in first and instance.goal_rc not in first
        altered = flip_cells(instance.map_design, first)
        assert is_reachable(altered, instance.start_rc, instance.goal_rc)


def test_sample_random_flips_rejects_bad_budgets(instance) -> None:
    """A non-positive or oversized budget is a programming error, not a draw."""
    with pytest.raises(ValueError, match="budget must be positive"):
        sample_random_flips(instance, 0, np.random.default_rng(0))
    with pytest.raises(ValueError, match="exceeds"):
        sample_random_flips(instance, 10_000, np.random.default_rng(0))


def test_sample_returns_none_when_nothing_is_admissible() -> None:
    """In a one-cell-wide corridor, every flip disconnects the goal."""
    from src.data import MazeInstance

    corridor = np.zeros((3, 5), dtype=np.float32)
    corridor[1, :] = 1.0
    inst = MazeInstance(
        instance_id=-1,
        split="test",
        map_design=corridor,
        goal_rc=(1, 4),
        start_rc=(1, 0),
        opt_dist=np.zeros_like(corridor, dtype=np.float64),
        start_seed=0,
    )
    # Only the corridor's interior cells and the walls exist; flipping a wall
    # to free is admissible, so restrict to free cells to force the dead end.
    assert sample_random_flips(inst, 3, np.random.default_rng(0), free_only=True) is None


def test_enumerate_k1_flips_matches_filtered_candidates(instance) -> None:
    """Enumeration is the candidate set minus the cells that seal the goal."""
    flips = enumerate_k1_flips(instance)
    assert all(len(cells) == 1 for cells in flips)
    flat = [cells[0] for cells in flips]
    assert len(set(flat)) == len(flat), "no duplicates"
    assert len(flips) <= config.MAP_SIZE**2 - 2
    for rc in flat:
        assert is_reachable(flip_cells(instance.map_design, [rc]), instance.start_rc, instance.goal_rc)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def test_baseline_matches_the_planner_on_the_original_map(pretrained_planner, instance) -> None:
    """`compute_baseline` reports the planner's own C0O0 numbers."""
    import torch

    from src.metrics import n_expansions, path_cost

    baseline = compute_baseline(pretrained_planner, instance)
    maps, starts, goals = instance.to_tensors()
    with torch.no_grad():
        out = pretrained_planner(maps, starts, goals)
    assert baseline.path_cost == path_cost(out.paths)[0]
    assert baseline.expansions == n_expansions(out.histories)[0]
    assert baseline.optimal_cost == instance.optimal_cost
    assert baseline.opt_ratio >= 1.0
    assert baseline.cost_map.shape == instance.map_design.shape


def test_double_flip_of_one_cell_is_a_no_op(pretrained_planner, instance) -> None:
    """Flipping a cell twice restores the map, so the row must equal the baseline.

    An identity check on the whole harness: map construction, encoder, search,
    solver and the ΔCost columns all have to agree that nothing changed.
    """
    cell = next(
        rc
        for rc in candidate_cells(instance.map_design, instance.start_rc, instance.goal_rc)
        if instance.map_design[rc] == config.FREE
    )
    request = AlterationRequest(instance=instance, flip_cells=(cell, cell))
    rows = evaluate_alterations(pretrained_planner, [request])
    row = rows[0]

    assert row["opt_C1O1"] == row["opt_C0O0"]
    assert row["exp_C1O1"] == row["exp_C0O0"]
    assert row["classical_opt_cost_altered"] == row["classical_opt_cost_original"]
    assert row["classical_expansions_altered"] == row["classical_expansions_original"]
    assert row["dcost_l1"] == 0.0
    assert np.isnan(row["dcost_frac_under_flips"])
    assert row["path_valid_C1O1"]


def test_evaluation_rows_are_self_consistent(pretrained_planner, test_split) -> None:
    """Rows carry a valid path, sane ratios and the right identifiers."""
    instances = [test_split.instance(i, seed=0) for i in range(4)]
    requests = []
    for inst in instances:
        cells = sample_random_flips(inst, 2, np.random.default_rng([0, inst.instance_id]))
        assert cells is not None
        requests.append(AlterationRequest(instance=inst, flip_cells=cells, draw=0))

    rows = evaluate_alterations(pretrained_planner, requests, batch_size=2)
    assert len(rows) == len(requests)
    for row, inst in zip(rows, instances):
        assert row["instance_id"] == inst.instance_id
        assert row["split"] == "test"
        assert row["budget"] == 2
        assert row["method"] == "random" and row["objective"] == "none"
        assert row["reachable"] is True
        assert row["path_valid_C1O1"], row["path_invalid_reason"]
        assert row["opt_C1O1"] >= 1.0, "a planner cannot beat the altered optimum"
        assert row["exp_C1O1"] > 0.0
        assert row["dcost_l1"] > 0.0
        assert 0.0 <= row["dcost_frac_under_flips"] <= 1.0
        assert row["n_obstacles_added"] + row["n_obstacles_removed"] == 2


def test_evaluation_refuses_unreachable_alterations(pretrained_planner) -> None:
    """A flip that seals the goal must be rejected, not silently scored.

    Built on a synthetic corridor rather than a dataset map on purpose: no
    single flip can disconnect the goal on these maps (0 of 9 068 tried, see
    `docs/verified.md` section 11), so a real instance cannot exercise the
    guard at k=1 and the test would silently skip.
    """
    from src.data import MazeInstance

    corridor = np.zeros((config.MAP_SIZE, config.MAP_SIZE), dtype=np.float32)
    corridor[1, :] = 1.0  # one open row, walls above and below
    inst = MazeInstance(
        instance_id=-1,
        split="test",
        map_design=corridor,
        goal_rc=(1, config.MAP_SIZE - 1),
        start_rc=(1, 0),
        opt_dist=np.zeros_like(corridor, dtype=np.float64),
        start_seed=0,
    )
    sealing = (1, config.MAP_SIZE // 2)
    assert not is_reachable(flip_cells(corridor, [sealing]), inst.start_rc, inst.goal_rc)

    request = AlterationRequest(instance=inst, flip_cells=(sealing,))
    with pytest.raises(ValueError, match="disconnect the goal"):
        evaluate_alterations(pretrained_planner, [request])


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def test_summarize_rows_separates_caused_damage_from_standing_error() -> None:
    """`frac_newly_*` conditions on the baseline; `frac_*` does not.

    Three synthetic k=1 alterations: one already-suboptimal instance that stays
    suboptimal, one optimal instance broken by the flip, one optimal instance
    that survives. The raw share is 2/3, but only 1 of the 2 instances that
    *could* be broken was.
    """
    rows = [
        {"budget": 1, "opt_C0O0": 1.2, "opt_C1O1": 1.2, "exp_C0O0": 0.5, "exp_C1O1": 0.5,
         "dcost_l1": 1.0, "dcost_frac_under_flips": 0.1,
         "classical_opt_cost_original": 20.0, "classical_opt_cost_altered": 20.0,
         "n_obstacles_added": 1, "n_obstacles_removed": 0},
        {"budget": 1, "opt_C0O0": 1.0, "opt_C1O1": 1.1, "exp_C0O0": 0.5, "exp_C1O1": 2.0,
         "dcost_l1": 2.0, "dcost_frac_under_flips": 0.2,
         "classical_opt_cost_original": 20.0, "classical_opt_cost_altered": 19.0,
         "n_obstacles_added": 0, "n_obstacles_removed": 1},
        {"budget": 1, "opt_C0O0": 1.0, "opt_C1O1": 1.0, "exp_C0O0": 0.5, "exp_C1O1": 0.6,
         "dcost_l1": 3.0, "dcost_frac_under_flips": 0.3,
         "classical_opt_cost_original": 20.0, "classical_opt_cost_altered": 20.0,
         "n_obstacles_added": 1, "n_obstacles_removed": 0},
    ]
    table = summarize_rows(rows)
    assert len(table) == 1
    entry = table[0]
    assert entry["n"] == 3
    assert abs(entry["frac_opt_deviating"] - 2 / 3) < 1e-12
    assert abs(entry["frac_newly_suboptimal"] - 0.5) < 1e-12
    assert abs(entry["frac_exp_above_threshold"] - 1 / 3) < 1e-12
    assert abs(entry["mean_dcost_l1"] - 2.0) < 1e-12


def test_summarize_rows_groups_by_budget() -> None:
    """Budgets are reported separately and in increasing order."""
    rows = [
        {"budget": b, "opt_C0O0": 1.0, "opt_C1O1": 1.0, "exp_C0O0": 0.5, "exp_C1O1": 0.5,
         "dcost_l1": 1.0, "dcost_frac_under_flips": 0.1,
         "classical_opt_cost_original": 20.0, "classical_opt_cost_altered": 20.0,
         "n_obstacles_added": 1, "n_obstacles_removed": 0}
        for b in (8, 1, 4, 2)
    ]
    table = summarize_rows(rows)
    assert [entry["budget"] for entry in table] == [1, 2, 4, 8]


def test_summarize_by_direction_splits_pillars_from_doors() -> None:
    """k=1 rows are split by whether the flip added or removed an obstacle."""
    rows = [
        {"budget": 1, "opt_C0O0": 1.0, "opt_C1O1": 1.0, "exp_C0O0": 0.5, "exp_C1O1": 0.5,
         "dcost_l1": 1.0, "dcost_frac_under_flips": 0.1,
         "classical_opt_cost_original": 20.0, "classical_opt_cost_altered": 20.0,
         "n_obstacles_added": 1, "n_obstacles_removed": 0},
        {"budget": 1, "opt_C0O0": 1.0, "opt_C1O1": 1.2, "exp_C0O0": 0.5, "exp_C1O1": 0.5,
         "dcost_l1": 3.0, "dcost_frac_under_flips": 0.1,
         "classical_opt_cost_original": 20.0, "classical_opt_cost_altered": 18.0,
         "n_obstacles_added": 0, "n_obstacles_removed": 1},
        {"budget": 2, "opt_C0O0": 1.0, "opt_C1O1": 1.0, "exp_C0O0": 0.5, "exp_C1O1": 0.5,
         "dcost_l1": 1.0, "dcost_frac_under_flips": 0.1,
         "classical_opt_cost_original": 20.0, "classical_opt_cost_altered": 20.0,
         "n_obstacles_added": 1, "n_obstacles_removed": 1},
    ]
    table = summarize_by_direction(rows, budget=1)
    assert [entry["direction"] for entry in table] == [
        "free->obstacle (pillar)",
        "obstacle->free (door)",
    ]
    assert table[0]["frac_newly_suboptimal"] == 0.0
    assert table[1]["frac_newly_suboptimal"] == 1.0
    assert all(abs(entry["share_of_draws"] - 0.5) < 1e-12 for entry in table)
