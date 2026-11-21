"""WarCraft 12x12: ground truth, the obstacle-free search, and tile edits.

Skipped when the dataset has not been downloaded (it is not versioned; see
`src.config.WARCRAFT_DIR`).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src import config

pytestmark = pytest.mark.skipif(
    not (config.WARCRAFT_DIR / "test_maps.npy").exists(),
    reason="WarCraft 12x12 data not downloaded",
)


@pytest.fixture(scope="module")
def test_data():
    from src.warcraft import load_warcraft_split

    return load_warcraft_split("test")


@pytest.fixture(scope="module")
def random_planner():
    """An untrained planner with the WarCraft architecture (plumbing tests only)."""
    from neural_astar.planner import NeuralAstar

    from src.warcraft import ENCODER_KW

    torch.manual_seed(0)
    planner = NeuralAstar(learn_obstacles=True, **ENCODER_KW)
    planner.eval()
    return planner


def test_weights_decode_to_the_five_terrain_costs(test_data) -> None:
    from src.warcraft import TERRAIN_COSTS

    assert set(np.unique(test_data.weights)) == set(TERRAIN_COSTS)


def test_dataset_paths_are_optimal_under_our_solver(test_data) -> None:
    """The exact solver reproduces the dataset's optimum on every test map."""
    from src.warcraft import is_connected_path, optimal_cost, path_map_cost

    for i in range(len(test_data)):
        w = test_data.weights[i]
        assert is_connected_path(test_data.paths[i])
        assert path_map_cost(test_data.paths[i], w) == pytest.approx(optimal_cost(w), abs=1e-9)


def test_through_costs_mark_every_optimal_path(test_data) -> None:
    from src.warcraft import optimal_cost, through_costs, tile_features

    for i in range(20):
        w = test_data.weights[i]
        assert through_costs(w).min() == pytest.approx(optimal_cost(w))
        on_path = tile_features(w)["on_optimal_path"] > 0
        assert on_path[test_data.paths[i] > 0].all()


def test_search_gets_an_all_ones_mask(test_data, random_planner) -> None:
    """With learned obstacles the search never sees the map: the mask is all ones.

    This is why the cost/obstacle swap has no obstacle half on WarCraft: our
    `plan` (which passes ones explicitly) must equal upstream's forward pass.
    """
    from src.warcraft import plan, to_tensors

    images = test_data.images[:8]
    _, paths, hists = plan(random_planner, images)
    with torch.no_grad():
        out = random_planner(*to_tensors(images))
    np.testing.assert_array_equal(paths, out.paths[:, 0].numpy())
    np.testing.assert_array_equal(hists, out.histories[:, 0].numpy())


def test_transplant_changes_only_the_target_tile(test_data) -> None:
    from src.warcraft import tile_slice, transplant_tiles

    img, w = test_data.images[0], test_data.weights[0]
    donor = test_data.images[1]
    new_img, new_w = transplant_tiles(img, w, [(3, 4)], [donor], [(5, 6)], [9.2])
    diff = np.any(new_img != img, axis=-1)
    rows, cols = tile_slice((3, 4))
    diff[rows, cols] = False
    assert not diff.any()
    np.testing.assert_array_equal(new_img[rows, cols], donor[tile_slice((5, 6))])
    assert new_w[3, 4] == 9.2
    new_w[3, 4] = w[3, 4]
    np.testing.assert_array_equal(new_w, w)


def test_score_of_the_reference_path_is_optimal(test_data) -> None:
    from src.warcraft import score

    w = test_data.weights[:5]
    paths = test_data.paths[:5]
    s = score(w, paths, paths, np.broadcast_to(np.ones((12, 12)), (5, 12, 12)))
    np.testing.assert_allclose(s["opt"], 1.0)
    assert s["valid"].all()
