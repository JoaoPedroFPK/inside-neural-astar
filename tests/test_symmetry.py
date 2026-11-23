"""Rotation symmetry (notebook 02): the geometry is exact, the control is symmetric."""

from __future__ import annotations

import numpy as np
import pytest

from src.exact_solver import solve_dijkstra
from src.grid import validate_path_map
from src.symmetry import (
    classical_rotations,
    evaluate_rotations,
    iou,
    rotate_instance,
    rotate_rc,
    summarize_rotations,
    symmetrised_cost_maps,
    unrotate,
)


@pytest.mark.parametrize("k", [-1, 0, 1, 2, 3, 5])
def test_rotate_rc_matches_numpy_rot90_on_a_non_square_array(k: int) -> None:
    """Every cell lands where np.rot90 puts it, for any k and any shape."""
    shape = (5, 8)
    marks = np.arange(40).reshape(shape)
    turned = np.rot90(marks, k)
    for r in range(shape[0]):
        for c in range(shape[1]):
            assert turned[rotate_rc((r, c), k, shape)] == marks[r, c]


def test_rotate_instance_keeps_the_problem(test_split) -> None:
    """Start and goal stay free, the optimum is unchanged, and unrotate inverts the turn."""
    inst = test_split.instance(7)
    for k in range(4):
        rot = rotate_instance(inst, k)
        assert rot.map_design[rot.start_rc] == 1 and rot.map_design[rot.goal_rc] == 1
        assert rot.optimal_cost == inst.optimal_cost
        assert solve_dijkstra(rot.map_design, rot.start_rc, rot.goal_rc).cost == inst.optimal_cost
        assert np.array_equal(unrotate(rot.map_design, k), inst.map_design)


def test_classical_control_is_rotation_invariant_in_length(test_split) -> None:
    """Classical A* finds a route of the optimal length on every copy (unit costs, admissible h)."""
    for i in range(0, 100, 10):
        rows = classical_rotations(test_split.instance(i))
        assert len({r.optimal_cost for r in rows}) == 1
        assert all(r.astar_cost == rows[0].optimal_cost for r in rows)


def test_iou_edge_cases() -> None:
    a = np.zeros((3, 3)); b = np.zeros((3, 3))
    assert iou(a, b) == 1.0
    a[0, 0] = 1; b[0, 0] = 1; b[1, 1] = 1
    assert iou(a, b) == 0.5


# ------------------------------------------------------------------ with the planner
def test_evaluate_rotations_returns_legal_routes_in_the_original_frame(pretrained_planner, test_split) -> None:
    """Rotated-back routes are legal on the original map; k = 0 is the plain planner."""
    inst = test_split.instance(3)
    results = evaluate_rotations(pretrained_planner, inst)
    for res in results:
        ok, reason = validate_path_map(res.path_map, inst.map_design, inst.start_rc, inst.goal_rc)
        assert ok, f"k={res.k}: {reason}"
        assert res.optimal_cost == inst.optimal_cost
    report = summarize_rotations(results, free=inst.map_design)
    assert report.rows[0]["route_iou"] == 1.0 and report.rows[0]["cost_mae"] == 0.0


def test_symmetrised_encoder_is_exactly_equivariant(pretrained_planner, test_split) -> None:
    """Averaging over the four turns removes the encoder's orientation gap (pipeline sanity check)."""
    inst = test_split.instance(11)
    results = evaluate_rotations(pretrained_planner, inst, cost_map_fn=symmetrised_cost_maps(pretrained_planner))
    report = summarize_rotations(results, free=inst.map_design)
    assert max(row["cost_max"] for row in report.rows) < 1e-5
