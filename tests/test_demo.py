"""The interactive demo: its state logic and its click/key wiring, run headless."""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from scripts.demo import DemoState, DemoView, explain, order_path  # noqa: E402


def test_order_path_walks_the_route_from_the_start() -> None:
    """A set of marked cells comes back as a chain of 8-neighbours from the start."""
    path = np.zeros((5, 5))
    for rc in [(0, 0), (1, 1), (2, 1), (3, 2), (4, 3)]:
        path[rc] = 1
    assert order_path(path, (0, 0)) == [(0, 0), (1, 1), (2, 1), (3, 2), (4, 3)]


def test_explain_attributes_damage_by_majority_share() -> None:
    """The network gets the blame when its costs alone reproduce half the change."""
    values = {"C0O0": 10, "C0O1": 10, "C1O0": 13, "C1O1": 14}
    assert "network" in explain("Route", values, "moves", 0.5, "{:+.0f}")[0]
    values = {"C0O0": 10, "C0O1": 11, "C1O0": 11, "C1O1": 14}
    assert "together" in explain("Route", values, "moves", 0.5, "{:+.0f}")[0]
    values = {"C0O0": 10, "C0O1": 10, "C1O0": 10, "C1O1": 10}
    assert "barely" in explain("Route", values, "moves", 0.5, "{:+.0f}")[0]


@pytest.fixture(scope="module")
def state(pretrained_planner):
    return DemoState(pretrained_planner)


def test_toggle_refuses_endpoints_and_round_trips(state) -> None:
    """Start and goal cannot be flipped; flipping a cell twice restores the clean map."""
    state.load_map(0)
    assert not state.toggle(state.instance.start_rc)
    clean_moves = state.view["moves"]
    cell = next(rc for rc in [(10, 10), (12, 12), (20, 20)] if state.toggle(rc))
    assert state.flips == [cell] and state.view["swap"] is not None
    assert state.toggle(cell) and state.flips == []
    assert state.view["moves"] == clean_moves
    state.undo()
    assert state.flips == [cell]


def test_route_is_never_shorter_than_the_optimum(state) -> None:
    """Opt >= 1 by definition; the verdict follows the project's error rule."""
    state.load_map(3)
    state.gradient_step("optimality")
    v = state.view
    assert v["opt"] >= 1 - 1e-9
    level, _ = state.verdict()
    assert (level == "fail") == (v["moves"] > v["optimal_moves"])


def test_clicks_and_keys_reach_the_state(state) -> None:
    """A click on the map flips the cell under it; `0` resets."""
    state.load_map(0)
    view = DemoView(state, interactive=True)
    view._on_click(SimpleNamespace(inaxes=view.ax_map, xdata=10.2, ydata=9.8))
    assert state.flips == [(10, 10)]
    view._on_key(SimpleNamespace(key="0"))
    assert state.flips == []
    view.plt.close(view.fig)
