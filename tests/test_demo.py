"""The browser demo: its state logic and its local HTTP API, run headless."""

from __future__ import annotations

import json
import threading
import urllib.request

import numpy as np
import pytest

from scripts.demo import Demo, DemoState, explain, make_server, order_path


def test_order_path_never_jumps_between_touching_but_distant_cells() -> None:
    """The drawn chain follows the route, start to goal, one 8-neighbour step at a time.

    A greedy walk from the start can jump between two marked cells that touch
    but are not consecutive, drawing a segment across a wall; the ordered
    chain here must have exactly as many steps as the shortest chain.
    """
    path = np.zeros((5, 5))
    for rc in [(0, 0), (1, 0), (2, 0), (2, 1), (1, 2), (0, 2)]:  # a U shape
        path[rc] = 1
    route = order_path(path, (0, 0), (0, 2))
    assert route[0] == (0, 0) and route[-1] == (0, 2)
    assert all(max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1 for a, b in zip(route, route[1:]))


def test_explain_attributes_damage_by_majority_share() -> None:
    """The network gets the blame when its costs alone reproduce half the change."""
    assert "network" in explain({"C0O0": 10, "C0O1": 10, "C1O0": 13, "C1O1": 14}, 0.5)
    assert "together" in explain({"C0O0": 10, "C0O1": 11, "C1O0": 11, "C1O1": 14}, 0.5)
    assert "barely" in explain({"C0O0": 10, "C0O1": 10, "C1O0": 10, "C1O1": 10}, 0.5)


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


def test_drawn_route_matches_the_planner_on_attacked_maps(state) -> None:
    """After attacks, the drawn route is legal and as long as the planner's path."""
    state.load_map(3)
    for _ in range(3):
        state.gradient_step("optimality")
        v = state.view
        route = v["path"]
        assert route[0] == state.instance.start_rc and route[-1] == state.instance.goal_rc
        assert all(v["map"][rc] == 1 for rc in route)
        assert len(route) - 1 == v["moves"]
    level, _ = state.verdict()
    assert (level == "fail") == (v["moves"] > v["optimal_moves"])


def test_http_api_serves_the_page_state_and_actions(state) -> None:
    """GET / is the page, GET /api/state the JSON, POST /api/action changes it."""
    state.load_map(0)
    demo = Demo(state)
    server = make_server(demo, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        assert b"<canvas" in urllib.request.urlopen(base + "/").read()
        snap = json.loads(urllib.request.urlopen(base + "/api/state?x=1").read())
        assert snap["mode"] == "maze" and len(snap["grid"]) == 32 and snap["flips"] == []
        request = urllib.request.Request(base + "/api/action", data=json.dumps(
            {"name": "flip", "arg": [10, 10]}).encode(), method="POST")
        snap = json.loads(urllib.request.urlopen(request).read())
        assert snap["flips"] == [[10, 10]]
        # A second tab switching the server's default map type must not
        # redirect this tab's maze clicks: the page names its map type.
        demo.mode = "warcraft"
        request = urllib.request.Request(base + "/api/action", data=json.dumps(
            {"name": "flip", "arg": [10, 10], "mode": "maze"}).encode(), method="POST")
        snap = json.loads(urllib.request.urlopen(request).read())
        assert snap["mode"] == "maze" and snap["flips"] == []
    finally:
        server.shutdown()
        server.server_close()
