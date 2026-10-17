"""PLAN.md step 4: the geometric feature library.

Each feature is checked on a hand-built map where the right answer can be read
off by eye, and then the whole library is checked for alignment against the
dataset on real instances.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import config
from src.data import ACTION_TO_MOVE, optimal_trajectory
from src.exact_solver import distance_field
from src.features import (
    AUX_NAMES,
    FEATURE_NAMES,
    TOPOLOGY_OPEN,
    WALL_ZONE_ADJACENT,
    WALL_ZONE_INTERIOR,
    WALL_ZONE_OBSTACLE,
    chebyshev_to_goal,
    compute_instance_features,
    corridor_span,
    corridor_width,
    dist_to_obstacle,
    euclidean_to_goal,
    obstacle_density,
    on_any_optimal_path,
    on_solver_path,
    topology_class,
    wall_zone,
)


@pytest.fixture()
def pillar_map():
    """A 5x7 open area with a single obstacle at (2, 3)."""
    m = np.ones((5, 7), dtype=np.float32)
    m[2, 3] = 0.0
    return m


# --------------------------------------------------------------------------- #
# Goal geometry
# --------------------------------------------------------------------------- #

def test_goal_distances_are_zero_at_the_goal() -> None:
    """Both goal metrics vanish at the goal and grow away from it."""
    goal = (2, 3)
    euclid = euclidean_to_goal((5, 7), goal)
    cheb = chebyshev_to_goal((5, 7), goal)
    assert euclid[goal] == 0.0 and cheb[goal] == 0.0
    assert cheb[0, 0] == max(abs(0 - 2), abs(0 - 3)) == 3
    assert abs(euclid[0, 0] - np.hypot(2, 3)) < 1e-12
    # Chebyshev never exceeds Euclidean on a grid, and is the 8-connected metric.
    assert np.all(cheb <= euclid + 1e-12)


def test_chebyshev_matches_the_upstream_heuristic_without_tie_break() -> None:
    """`chebyshev_to_goal` is exactly upstream's `h` minus 0.001 * Euclidean."""
    from src.exact_solver import chebyshev_heuristic

    goal = (7, 19)
    shape = (config.MAP_SIZE, config.MAP_SIZE)
    recovered = chebyshev_heuristic(shape, goal) - 0.001 * euclidean_to_goal(shape, goal)
    assert np.allclose(recovered, chebyshev_to_goal(shape, goal))


# --------------------------------------------------------------------------- #
# Obstacle geometry
# --------------------------------------------------------------------------- #

def test_dist_to_obstacle_counts_the_grid_edge_as_blocked(pillar_map) -> None:
    """Border cells are distance 1 — outside the grid is an obstacle."""
    chebyshev = dist_to_obstacle(pillar_map, "chebyshev")
    assert chebyshev[0, 0] == 1.0, "border cell must be adjacent to 'outside'"
    assert chebyshev[2, 3] == 0.0, "the obstacle itself is 0"
    assert chebyshev[2, 2] == 1.0, "neighbour of the pillar"
    assert chebyshev[1, 1] == 2.0
    euclidean = dist_to_obstacle(pillar_map, "euclidean")
    assert euclidean[2, 2] == 1.0
    assert abs(euclidean[1, 1] - np.sqrt(2)) < 1e-9 or euclidean[1, 1] == 2.0


def test_dist_to_obstacle_rejects_unknown_metric(pillar_map) -> None:
    """A typo in the metric name must fail loudly."""
    with pytest.raises(ValueError, match="unknown metric"):
        dist_to_obstacle(pillar_map, "manhattan")


def test_obstacle_density_windows(pillar_map) -> None:
    """Density is the obstacle fraction of the window, edges included."""
    density3 = obstacle_density(pillar_map, 3)
    assert abs(density3[2, 2] - 1 / 9) < 1e-12, "one pillar in an interior 3x3"
    assert abs(density3[0, 0] - 5 / 9) < 1e-12, "corner: 5 of 9 cells are outside"
    density5 = obstacle_density(pillar_map, 5)
    assert abs(density5[2, 3] - 1 / 25) < 1e-12
    with pytest.raises(ValueError, match="odd"):
        obstacle_density(pillar_map, 4)


def test_corridor_width_is_clearance_not_width(pillar_map) -> None:
    """`corridor_width_min4` is the minimum clearance over four directions."""
    width = corridor_width(pillar_map)
    assert width[2, 3] == 0.0, "obstacles get 0"
    assert width[0, 0] == 1.0, "a corner touches the edge in two directions"
    assert width[1, 1] == 2.0
    assert width[2, 2] == 1.0, "next to the pillar"


def test_corridor_span_measures_the_free_line_through_a_cell(pillar_map) -> None:
    """`corridor_span_min` is the narrower of the two free spans through a cell."""
    span = corridor_span(pillar_map)
    assert span[0, 0] == 5.0, "7 wide horizontally, 5 tall vertically -> 5"
    assert span[2, 0] == 3.0, "its row is cut by the pillar at column 3"
    assert span[1, 3] == 2.0, "its column is cut by the pillar below"
    assert span[2, 3] == 0.0


def test_single_gap_in_a_wall_has_span_one() -> None:
    """A one-cell doorway is the narrowest possible span."""
    m = np.ones((5, 5), dtype=np.float32)
    m[2, :] = 0.0
    m[2, 2] = 1.0  # a doorway through the wall
    assert corridor_span(m)[2, 2] == 1.0
    assert corridor_width(m)[2, 2] == 1.0


# --------------------------------------------------------------------------- #
# Partitions
# --------------------------------------------------------------------------- #

def test_partitions_label_obstacles_and_open_space(pillar_map) -> None:
    """Topology and wall-zone codes behave on a known map."""
    topo = topology_class(pillar_map)
    zone = wall_zone(pillar_map)
    assert topo[2, 3] == -1 and zone[2, 3] == WALL_ZONE_OBSTACLE
    assert topo[1, 1] == TOPOLOGY_OPEN, "all eight neighbours free"
    assert zone[0, 0] == WALL_ZONE_ADJACENT
    assert zone[1, 1] == 2  # WALL_ZONE_NEAR
    big = np.ones((9, 9), dtype=np.float32)
    assert wall_zone(big)[4, 4] == WALL_ZONE_INTERIOR


# --------------------------------------------------------------------------- #
# Path indicators
# --------------------------------------------------------------------------- #

def test_on_any_optimal_path_is_the_union_of_optimal_routes() -> None:
    """In an open 3x3, every cell of every monotone diagonal route qualifies."""
    m = np.ones((3, 3), dtype=np.float32)
    indicator, d_start, d_goal = on_any_optimal_path(m, (0, 0), (2, 2))
    assert indicator[0, 0] == 1.0 and indicator[2, 2] == 1.0
    assert indicator[1, 1] == 1.0, "the diagonal midpoint is on an optimal path"
    assert indicator[0, 2] == 0.0, "a corner detour costs more"
    assert d_start[2, 2] == 2.0 and d_goal[0, 0] == 2.0


def test_solver_path_is_a_subset_of_any_optimal_path(test_split) -> None:
    """The specific path the solver returns lies inside the optimal-path set."""
    for i in range(3):
        inst = test_split.instance(i, seed=0)
        indicator, _, _ = on_any_optimal_path(
            inst.map_design, inst.start_rc, inst.goal_rc, inst.optimal_cost
        )
        specific = on_solver_path(inst.map_design, inst.start_rc, inst.goal_rc)
        assert np.all(indicator[specific > 0] == 1.0)
        assert specific.sum() == inst.optimal_cost + 1  # cells = moves + 1


def test_reference_trajectory_is_an_optimal_path(test_split) -> None:
    """The dataset's policy walk is optimal and lies in the optimal-path set.

    This is the training target (`L1(histories, opt_trajs)`), so if it were not
    optimal the supervision itself would be inconsistent with the solver.
    """
    for i in range(5):
        inst = test_split.instance(i, seed=0)
        traj = test_split.reference_trajectory(i, inst.start_rc)
        indicator, _, _ = on_any_optimal_path(
            inst.map_design, inst.start_rc, inst.goal_rc, inst.optimal_cost
        )
        assert traj[inst.start_rc] == 1.0, "the start is marked"
        assert traj[inst.goal_rc] == 0.0, "upstream does not mark the goal"
        assert traj.sum() == inst.optimal_cost, "cells = moves (goal unmarked)"
        assert np.all(indicator[traj > 0] == 1.0)


def test_reference_trajectory_rejects_a_broken_policy() -> None:
    """A policy that walks in circles raises instead of looping forever."""
    policy = np.zeros((8, 1, 4, 4), dtype=np.float32)
    policy[1, 0, 0, 0] = 1.0  # (0,0) -> right
    policy[2, 0, 0, 1] = 1.0  # (0,1) -> left, back again
    with pytest.raises(ValueError, match="revisits"):
        optimal_trajectory(policy, (0, 0), (3, 3))


def test_action_table_matches_upstream() -> None:
    """The action->move table is upstream's, in upstream's order."""
    assert ACTION_TO_MOVE == (
        (-1, 0), (0, 1), (0, -1), (1, 0), (-1, 1), (-1, -1), (1, 1), (1, -1)
    )


# --------------------------------------------------------------------------- #
# Assembly and alignment
# --------------------------------------------------------------------------- #

def test_compute_instance_features_returns_every_array(test_split) -> None:
    """All declared arrays are present, correctly shaped and typed."""
    inst = test_split.instance(0, seed=0)
    cost = np.full(inst.map_design.shape, 0.25, dtype=np.float32)
    reference = test_split.reference_trajectory(0, inst.start_rc)
    out = compute_instance_features(inst, cost_map=cost, reference_path=reference)

    assert set(out) == set(FEATURE_NAMES) | set(AUX_NAMES)
    for name in FEATURE_NAMES:
        assert out[name].shape == inst.map_design.shape
        assert out[name].dtype == np.float32
    assert out["topology_class"].dtype == np.int8
    assert out["wall_zone"].dtype == np.int8
    assert np.array_equal(out["cost_map"], cost)
    assert out["is_start"][inst.start_rc] == 1.0 and out["is_start"].sum() == 1
    assert out["is_goal"][inst.goal_rc] == 1.0 and out["is_goal"].sum() == 1


def test_geodesic_features_match_the_dataset(test_split) -> None:
    """`geodesic_to_goal` is the dataset's field; `geodesic_from_start` agrees with it.

    The alignment check that matters: a transposed or shifted feature would
    break this immediately.
    """
    for i in range(3):
        inst = test_split.instance(i, seed=0)
        out = compute_instance_features(inst)
        free = inst.map_design == config.FREE
        both = free & np.isfinite(inst.opt_dist) & np.isfinite(out["geodesic_to_goal"])
        assert np.array_equal(out["geodesic_to_goal"][both], inst.opt_dist[both].astype(np.float32))
        assert out["geodesic_from_start"][inst.start_rc] == 0.0
        assert out["geodesic_to_goal"][inst.goal_rc] == 0.0
        # The two fields agree on the optimum through the start.
        assert out["geodesic_to_goal"][inst.start_rc] == np.float32(inst.optimal_cost)


def test_unreachable_cells_are_infinite_not_sentinels(test_split) -> None:
    """Free cells outside the goal's component get inf, and are masked out."""
    inst = test_split.instance(0, seed=0)
    out = compute_instance_features(inst)
    free = out["is_free"] > 0
    unreachable = free & ~np.isfinite(out["geodesic_to_goal"])
    assert unreachable.any(), "instance 0 has free cells with no path to the goal"
    assert np.all(out["reachable"][unreachable] == 0.0)
    assert np.all(np.isinf(out["geodesic_to_goal"][unreachable]))


def test_features_are_deterministic(test_split) -> None:
    """Two computations of the same instance agree bit for bit."""
    inst = test_split.instance(1, seed=0)
    first = compute_instance_features(inst)
    second = compute_instance_features(inst)
    for name in FEATURE_NAMES:
        assert np.array_equal(
            np.nan_to_num(first[name], posinf=-1), np.nan_to_num(second[name], posinf=-1)
        )
