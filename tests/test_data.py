"""Dataset facts this project builds on .

If any of these fail, `docs/verified.md` is out of date and every downstream
result is suspect.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import config
from src.grid import bfs_distances, flip_cells, is_reachable, local_topology_class


def test_split_shapes_and_sizes(test_split) -> None:
    """The test split has 100 instances of 32x32 with one-hot goals."""
    assert len(test_split) == config.SPLIT_SIZES["test"]
    assert test_split.map_designs.shape == (100, config.MAP_SIZE, config.MAP_SIZE)
    assert test_split.goal_maps.shape == (100, 1, config.MAP_SIZE, config.MAP_SIZE)
    assert test_split.opt_dists.shape == (100, 1, config.MAP_SIZE, config.MAP_SIZE)
    assert np.all(test_split.goal_maps.sum(axis=(1, 2, 3)) == 1)


def test_map_polarity_is_one_means_free(test_split) -> None:
    """Maps are binary with 1 = free; the goal always sits on a free cell.

    This is the premise behind using the map as an obstacle *mask*: the A*
    module multiplies candidate neighbours by it
    (`differentiable_astar.py:229`), so 1 must mean traversable.
    """
    maps = test_split.map_designs
    assert set(np.unique(maps).tolist()) == {config.OBSTACLE, config.FREE}
    for i in range(len(test_split)):
        assert maps[i][test_split.goal_rc(i)] == config.FREE


def test_stored_distances_are_unit_cost_8_connected(test_split) -> None:
    """Dataset distances equal unit-cost 8-connected BFS from the goal, exactly.

    This is the fact the whole cost model rests on: diagonal moves cost 1, not
    sqrt(2) (`planning-datasets/.../dijkstra.py:153`). Checked on every
    instance of the test split, on every free cell.
    """
    for i in range(len(test_split)):
        map_design = test_split.map_designs[i]
        goal_rc = test_split.goal_rc(i)
        stored = test_split.opt_dists[i, 0]
        bfs = bfs_distances(map_design, goal_rc)

        free = map_design == config.FREE
        # Reachable free cells: exact agreement.
        both_finite = free & np.isfinite(bfs) & np.isfinite(stored)
        assert np.array_equal(bfs[both_finite], stored[both_finite]), (
            f"instance {i}: stored distances disagree with unit-cost BFS"
        )
        # And the two notions of "unreachable" coincide on free cells.
        assert np.array_equal(np.isfinite(bfs) & free, np.isfinite(stored) & free), (
            f"instance {i}: disagreement on which free cells are reachable"
        )
        # Obstacles are always sentinel/inf, never a finite distance.
        assert not np.isfinite(stored[~free]).any()


def test_unreachable_free_cells_exist_in_mazes(test_split) -> None:
    """Some free cells are outside the goal's component — starts must avoid them.

    Upstream samples goals from the largest connected component only
    (`generate_spp_instances.py`, `from_largest=True`), so mazes routinely
    contain free cells with no path to the goal. Any start-selection or
    flip-reachability logic that ignores this would silently produce
    unsolvable instances.
    """
    free = test_split.map_designs == config.FREE
    unreachable_free = free & ~np.isfinite(test_split.opt_dists[:, 0])
    assert unreachable_free.any(), "expected at least one unreachable free cell"


def test_fixed_start_is_deterministic_and_solvable(test_split) -> None:
    """The fixed start is reproducible, reachable, and not the goal."""
    for i in (0, 7, 42, 99):
        a = test_split.instance(i, seed=0)
        b = test_split.instance(i, seed=0)
        assert a.start_rc == b.start_rc
        assert a.start_rc != a.goal_rc
        assert a.map_design[a.start_rc] == config.FREE
        assert np.isfinite(a.optimal_cost) and a.optimal_cost > 0
        assert is_reachable(a.map_design, a.start_rc, a.goal_rc)


def test_fixed_start_depends_on_seed_and_instance(test_split) -> None:
    """Different seeds give different draws; different instances are independent."""
    starts_seed0 = [test_split.instance(i, seed=0).start_rc for i in range(20)]
    starts_seed1 = [test_split.instance(i, seed=1).start_rc for i in range(20)]
    assert starts_seed0 != starts_seed1


def test_farthest_start_rule(test_split) -> None:
    """The `farthest` rule returns a cell at the maximum finite distance."""
    inst = test_split.instance(3, rule="farthest")
    finite = inst.opt_dist[np.isfinite(inst.opt_dist)]
    assert inst.optimal_cost == finite.max()


def test_start_band_matches_upstream_protocol(test_split) -> None:
    """Fixed starts fall in the same distance band upstream samples from.

    Upstream draws starts from the 55-100th percentile of distance-to-goal
    (`neural_astar/utils/data.py:200-217`); staying in that band keeps our
    fixed starts in-distribution for the pretrained model.
    """
    from src.data import START_PERCENTILE_BAND

    for i in range(len(test_split)):
        inst = test_split.instance(i, seed=0)
        reachable = inst.opt_dist[np.isfinite(inst.opt_dist) & (inst.opt_dist > 0)]
        low = np.percentile(reachable, START_PERCENTILE_BAND[0])
        assert inst.optimal_cost >= low


def test_flip_cells_is_pure_and_invertible() -> None:
    """Flipping is a pure function and flipping twice restores the map."""
    rng = np.random.default_rng(0)
    m = (rng.random((8, 8)) > 0.3).astype(np.float32)
    original = m.copy()
    cells = [(0, 0), (3, 4), (7, 7)]
    flipped = flip_cells(m, cells)
    assert np.array_equal(m, original), "flip_cells must not modify its input"
    assert np.array_equal(flip_cells(flipped, cells), original)
    for rc in cells:
        assert flipped[rc] == 1 - original[rc]
    with pytest.raises(ValueError):
        flip_cells(m, [(8, 0)])


def test_reachability_rejects_sealed_goal() -> None:
    """Sealing the goal's only exit must be detected as unreachable."""
    m = np.zeros((5, 5), dtype=np.float32)
    m[2, :] = 1.0  # a single horizontal corridor
    assert is_reachable(m, (2, 0), (2, 4))
    sealed = flip_cells(m, [(2, 2)])
    assert not is_reachable(sealed, (2, 0), (2, 4))


def test_local_topology_counts_free_neighbours() -> None:
    """Topology class is the count of free Moore neighbours; obstacles get -1."""
    m = np.zeros((3, 3), dtype=np.float32)
    m[1, 1] = 1.0
    topo = local_topology_class(m)
    assert topo[1, 1] == 0
    assert topo[0, 0] == -1
    m_full = np.ones((3, 3), dtype=np.float32)
    assert local_topology_class(m_full)[1, 1] == 8
    assert local_topology_class(m_full)[0, 0] == 3


def test_all_map_families_are_present_and_loadable() -> None:
    """The eight 32x32 families ship with the nested submodule and load.

    They are the validity check of plan step 7b: the step-3 and step-4 findings
    were measured on one geometry, and these cost nothing to test against.
    """
    from src.data import load_split

    assert len(config.MAP_FAMILIES) == 8
    assert "mazes" in config.MAP_FAMILIES and "forest" in config.MAP_FAMILIES
    for name in config.MAP_FAMILIES:
        path = config.family_npz(name)
        assert path.exists(), f"{name}: {path} missing"
    # Loading one other family must give the same shape and conventions.
    forest = load_split("test", npz_path=config.family_npz("forest"))
    assert len(forest) == config.SPLIT_SIZES["test"]
    assert forest.map_designs.shape[1:] == (config.MAP_SIZE, config.MAP_SIZE)
    assert set(np.unique(forest.map_designs).tolist()) == {config.OBSTACLE, config.FREE}


def test_family_registry_rejects_unknown_names() -> None:
    """A typo in a family name fails loudly rather than building a bad path."""
    with pytest.raises(KeyError):
        config.family_npz("labyrinth")
