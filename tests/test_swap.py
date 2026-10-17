"""PLAN.md step 7: the 2x2 cost/obstacle swap harness."""

from __future__ import annotations

import numpy as np
import pytest

from src.data import MazeInstance
from src.swap import (
    CONDITIONS,
    SwapResult,
    classify_route,
    run_swap,
    summarize_swaps,
    swap_columns,
)


def _result(opt_values, exp_values=None) -> SwapResult:
    """A hand-made `SwapResult` with the given per-condition Opt ratios."""
    labels = ("C0O0", "C0O1", "C1O0", "C1O1")
    exp_values = exp_values or dict(zip(labels, [1.0, 1.0, 1.0, 1.0]))
    return SwapResult(
        instance_id=0,
        flip_cells=((1, 1),),
        opt=dict(zip(labels, opt_values)) if not isinstance(opt_values, dict) else opt_values,
        exp=exp_values,
        path_cost=dict(zip(labels, [10.0] * 4)),
        expansions=dict(zip(labels, [20.0] * 4)),
        valid=dict(zip(labels, [True] * 4)),
        dcost_l1=1.0,
        dcost_frac_under_flips=0.1,
        dcost_frac_outside_receptive_field=0.0,
    )


@pytest.fixture(scope="module")
def corridor_instance():
    """A synthetic map where opening a second doorway shortens the optimum."""
    grid = np.ones((9, 9), dtype=np.float32)
    grid[4, :] = 0.0       # a wall across the middle
    grid[4, 8] = 1.0       # the only way through, at the far right
    return MazeInstance(
        instance_id=-1,
        split="test",
        map_design=grid,
        goal_rc=(8, 0),
        start_rc=(0, 0),
        opt_dist=np.zeros((9, 9), dtype=np.float64),
        start_seed=0,
    )


# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #

def test_conditions_are_in_the_canonical_order() -> None:
    """The four conditions use the project's naming and ordering."""
    assert [label for label, _, _ in CONDITIONS] == ["C0O0", "C0O1", "C1O0", "C1O1"]
    assert CONDITIONS[0][1:] == (False, False)
    assert CONDITIONS[1][1:] == (False, True)
    assert CONDITIONS[2][1:] == (True, False)
    assert CONDITIONS[3][1:] == (True, True)


def test_mediation_arithmetic() -> None:
    """Mediation fractions follow the project definition."""
    result = _result([1.0, 1.1, 1.3, 1.4])
    fractions = result.mediation("opt")
    assert abs(fractions["encoder"] - 0.75) < 1e-12   # (1.3-1.0)/(1.4-1.0)
    assert abs(fractions["search"] - 0.25) < 1e-12    # (1.1-1.0)/(1.4-1.0)
    assert abs(fractions["total_effect"] - 0.4) < 1e-12
    with pytest.raises(ValueError, match="unknown metric"):
        result.mediation("hmean")


def test_mediation_is_nan_when_nothing_broke() -> None:
    """No damage means no decomposition, reported as nan rather than 0."""
    fractions = _result([1.0, 1.0, 1.0, 1.0]).mediation("opt")
    assert np.isnan(fractions["encoder"]) and np.isnan(fractions["search"])


def test_route_classification_covers_every_pattern() -> None:
    """Each mediation pattern gets its label."""
    assert classify_route(_result([1.0, 1.0, 1.0, 1.0]).mediation("opt")) == "no damage"
    assert classify_route(_result([1.0, 1.0, 1.4, 1.4]).mediation("opt")) == "encoder"
    assert classify_route(_result([1.0, 1.4, 1.0, 1.4]).mediation("opt")) == "search"
    assert classify_route(_result([1.0, 1.4, 1.4, 1.4]).mediation("opt")) == "both"
    assert classify_route(_result([1.0, 1.0, 1.0, 1.4]).mediation("opt")) == "interaction"


def test_an_improvement_is_not_damage() -> None:
    """A flip that makes the planner *better* has nothing to decompose.

    Dividing by a negative total effect would flip the sign of both routes and
    produce confident nonsense, so these alterations are excluded from the
    mediation averages rather than being scored.
    """
    improved = _result([1.2, 1.1, 1.1, 1.0])
    assert classify_route(improved.mediation("opt")) == "no damage"
    summary = summarize_swaps([improved], "opt")
    assert summary["n_with_damage"] == 0
    assert np.isnan(summary["encoder_median"])


def test_summary_counts_and_averages() -> None:
    """The summary counts labels and averages only the damaged alterations."""
    results = [
        _result([1.0, 1.0, 1.4, 1.4]),   # encoder
        _result([1.0, 1.4, 1.0, 1.4]),   # search
        _result([1.0, 1.0, 1.0, 1.0]),   # no damage
    ]
    summary = summarize_swaps(results, "opt")
    assert summary["n"] == 3
    assert summary["n_with_damage"] == 2
    assert summary["labels"] == {"encoder": 1, "search": 1, "no damage": 1}
    assert abs(summary["encoder_mean"] - 0.5) < 1e-12   # mean of 1.0 and 0.0
    assert abs(summary["search_mean"] - 0.5) < 1e-12


def test_swap_columns_are_manifest_ready() -> None:
    """Flattening produces one column per condition plus the diagnostics."""
    columns = swap_columns(_result([1.0, 1.1, 1.3, 1.4]))
    for label in ("C0O0", "C0O1", "C1O0", "C1O1"):
        assert f"opt_{label}" in columns and f"exp_{label}" in columns
    assert "mediation_opt_encoder" in columns and "mediation_exp_search" in columns
    assert columns["dcost_frac_outside_receptive_field"] == 0.0


# --------------------------------------------------------------------------- #
# Against the real planner
# --------------------------------------------------------------------------- #

def test_a_no_op_alteration_leaves_all_four_conditions_identical(
    pretrained_planner, test_split
) -> None:
    """Flipping a cell twice restores the map, so every condition must agree."""
    inst = test_split.instance(0, seed=0)
    cell = (14, 17)
    result = run_swap(pretrained_planner, inst, [cell, cell])
    assert len(set(result.opt.values())) == 1
    assert len(set(result.exp.values())) == 1
    assert result.dcost_l1 == 0.0
    assert np.isnan(result.mediation("opt")["encoder"])


def test_every_condition_returns_a_legal_path(pretrained_planner, test_split) -> None:
    """Each condition's path must respect the mask it was actually given."""
    inst = test_split.instance(0, seed=0)
    result = run_swap(pretrained_planner, inst, [(14, 17)])
    assert all(result.valid.values()), result.valid


def test_cost_change_stays_inside_the_receptive_field(
    pretrained_planner, test_split
) -> None:
    """A k=1 flip cannot move the cost map beyond radius 5 of the flipped cell.

    If this ever fails, the encoder-route bound used throughout the mediation
    analysis is invalid, so it is asserted on every swap rather than assumed.
    """
    inst = test_split.instance(1, seed=0)
    result = run_swap(pretrained_planner, inst, [(12, 12)])
    assert result.dcost_frac_outside_receptive_field == 0.0
    assert result.dcost_l1 > 0.0


def test_reference_follows_the_obstacle_mask_not_the_cost_map(
    pretrained_planner, corridor_instance
) -> None:
    """Conditions on different masks are scored against different optima.

    Opening a second doorway genuinely shortens the optimal path, so the
    altered-mask conditions face a harder yardstick. If the reference were
    taken from the cost map instead, C0O1 would be scored against the old,
    longer optimum and would look falsely good.
    """
    from src.exact_solver import solve_dijkstra
    from src.grid import flip_cells

    door = (4, 0)
    original_optimum = solve_dijkstra(
        corridor_instance.map_design, corridor_instance.start_rc, corridor_instance.goal_rc
    ).cost
    altered_optimum = solve_dijkstra(
        flip_cells(corridor_instance.map_design, [door]),
        corridor_instance.start_rc,
        corridor_instance.goal_rc,
    ).cost
    assert altered_optimum < original_optimum, "the new doorway must shorten the route"

    result = run_swap(pretrained_planner, corridor_instance, [door])
    # Same raw path cost scored against two different optima must give two
    # different ratios; verify by reconstructing the ratio from the raw values.
    for label, optimum in (("C0O0", original_optimum), ("C1O1", altered_optimum)):
        assert abs(result.opt[label] - result.path_cost[label] / optimum) < 1e-9


def test_disconnecting_flips_are_rejected(pretrained_planner, corridor_instance) -> None:
    """Sealing the only doorway is refused rather than silently scored."""
    with pytest.raises(ValueError, match="disconnect the goal"):
        run_swap(pretrained_planner, corridor_instance, [(4, 8)])


def test_raw_mediation_uses_counts_not_ratios() -> None:
    """The raw variant decomposes path length directly, dodging the denominator.

    The ratio metrics divide by the altered map's own optimum, which can differ
    between conditions; the raw variant cannot, so agreement between the two is
    evidence that the normalisation is not driving the result.
    """
    labels = ("C0O0", "C0O1", "C1O0", "C1O1")
    result = SwapResult(
        instance_id=0,
        flip_cells=((1, 1),),
        opt=dict(zip(labels, [1.0, 1.0, 1.0, 1.1])),
        exp=dict(zip(labels, [1.0, 1.0, 1.0, 1.0])),
        path_cost=dict(zip(labels, [20.0, 20.0, 24.0, 28.0])),
        expansions=dict(zip(labels, [30.0, 30.0, 30.0, 30.0])),
        valid=dict(zip(labels, [True] * 4)),
        dcost_l1=1.0,
        dcost_frac_under_flips=0.1,
        dcost_frac_outside_receptive_field=0.0,
    )
    raw = result.mediation_raw("opt")
    assert abs(raw["encoder"] - 0.5) < 1e-12      # (24-20)/(28-20)
    assert abs(raw["search"]) < 1e-12
    assert abs(raw["total_effect"] - 8.0) < 1e-12
    assert np.isnan(result.mediation_raw("exp")["encoder"])
    with pytest.raises(ValueError, match="unknown metric"):
        result.mediation_raw("hmean")
