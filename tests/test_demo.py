"""The browser demo: its state logic and its local HTTP API, run headless."""

from __future__ import annotations

import json
import threading
import urllib.request

import numpy as np
import pytest

from scripts.demo import Demo, DemoState, explain, json_safe, make_server, order_path, snapshot


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


def test_rotation_turns_the_flips_with_the_map_and_keeps_the_optimum(state) -> None:
    """Turning keeps the problem: same optimum, flips carried along, and back again."""
    from src.symmetry import rotate_rc

    state.load_map(0)
    optimum = state.view["optimal_moves"]
    cell = next(rc for rc in [(10, 10), (12, 12), (20, 20)] if state.toggle(rc))
    state.rotate(1)
    assert state.rotation == 1 and state.flips == [rotate_rc(cell, 1, (32, 32))]
    state.rotate(0)
    assert state.flips == [cell]
    state.reset()
    for k in range(4):
        state.rotate(k)
        assert state.view["optimal_moves"] == optimum
        sym = state.view["symmetry"]
        assert [r["degrees"] for r in sym["rows"]] == [0, 90, 180, 270]
        assert sum(r["current"] for r in sym["rows"]) == 1
    json.dumps(json_safe(snapshot(state)), allow_nan=False)


def test_sabotage_on_a_shortcut_map_keeps_the_gate_open_and_optimal(state) -> None:
    """Each sabotage click adds one flip that leaves the gate open and still the shortest way."""
    state.load_scenario(0)
    assert state.view["sabotage"]["planner_uses_gate"] and not state.view["sabotage"]["success"]
    for n in (1, 2):
        state.sabotage_step()
        s = state.view["sabotage"]
        assert len(state.flips) == n and s["gate_open"] >= 1 and s["shortcut_optimal"]
    state.rotate(2)  # the gate turns with the board
    s = state.view["sabotage"]
    assert all(state.view["map"][tuple(rc)] in (0, 1) for rc in s["gate"]) and s["shortcut_optimal"]
    snap = json_safe(snapshot(state))
    assert snap["scenario"]["index"] == 0 and snap["rotation"] == 180
    json.dumps(snap, allow_nan=False)
    state.worst("optimality")
    assert "shortcut" in state.message
    state.load_map(0)
    assert state.scenario_idx is None and state.view["sabotage"] is None


@pytest.fixture(scope="module")
def explained(pretrained_planner, test_split):
    from scripts.demo_explain import MazeExplainer

    return DemoState(pretrained_planner, explainer=MazeExplainer(pretrained_planner, test_split))


def test_live_surrogates_reproduce_the_sufficiency_test(explained, test_split) -> None:
    """Planning with the live surrogates on the clean test maps gives step 5's figures.

    Step 5 stored 0.620 exactly optimal for the geometry surrogate and 0.920
    for the all-features one, and the guidance-strength control (the network's
    map weakened to ``0.5 + a * (cost - 0.5)``) at 0.83 / 0.84 / 0.84 / 0.86 /
    0.88 / 0.81 for a = 1 / 0.75 / 0.5 / 0.25 / 0.1 / 0; the demo recomputes all
    of them per map, so any drift shows up here.
    """
    optimal = {"geometric_h1": 0, "all": 0}
    strength = [0] * 6
    for i in range(len(test_split)):
        explained.load_map(i)
        view = explained.view
        for group, s in view["explain"]["surrogates"].items():
            optimal[group] += s["moves"] <= view["optimal_moves"]
        curve = view["explain"]["strength"]
        assert (curve[0]["moves"], curve[0]["expanded"]) == (view["moves"], view["expanded"])  # a = 1 is the network
        for k, point in enumerate(curve):
            strength[k] += point["moves"] <= view["optimal_moves"]
    assert optimal == {"geometric_h1": 62, "all": 92}
    assert strength == [83, 84, 84, 86, 88, 81]


def test_attribution_respects_the_receptive_field_and_refuses_walls(explained) -> None:
    """Integrated gradients credit only walls within radius 5; walls are not explained."""
    explained.load_map(0)
    grid = explained.view["map"]
    wall = tuple(int(v) for v in np.argwhere(grid == 0)[0])
    explained.attribute(wall)
    assert explained.view["attribution"] is None and "wall" in explained.message
    # a free cell next to a wall, so there is something to credit
    free = next((r, c) for r, c in np.argwhere(grid == 1)
                if grid[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2].min() == 0)
    explained.attribute(free)
    a = explained.view["attribution"]
    rows, cols = np.indices(grid.shape)
    outside = np.maximum(np.abs(rows - free[0]), np.abs(cols - free[1])) > 5
    assert np.all(a["ig"][outside] == 0) and np.all(a["ig"][grid == 1] == 0)
    assert a["ig_inside"] > 0.999 and a["completeness_error"] < 0.05
    # saliency is a gradient too, so it cannot reach past the receptive field either
    assert np.all(a["saliency"][outside] == 0) and a["saliency_inside"] > 0.999
    # a change to the map makes the attribution stale; the target is kept
    explained.toggle(next(rc for rc in [(20, 20), (21, 21), (25, 25)] if grid[rc] == 1))
    assert explained.view["attribution"] is None and explained.attr_target == free


def test_json_safe_turns_undefined_statistics_into_null() -> None:
    """The page must receive valid JSON: NaN and infinities become null."""
    from scripts.demo import json_safe

    out = json_safe({"a": float("nan"), "b": [1.0, float("inf")], "c": (2, {"d": -float("inf")})})
    assert out == {"a": None, "b": [1.0, None], "c": [2, {"d": None}]}
    json.dumps(out, allow_nan=False)


def test_page_feature_list_matches_the_surrogates_feature_groups() -> None:
    """The page's feature box lists as many features per surrogate as the fits use."""
    import re
    from pathlib import Path

    from src.surrogate import FEATURE_GROUPS

    page = (Path(__file__).resolve().parents[1] / "scripts" / "demo_page.html").read_text()
    block = page.split("const SURROGATE_FEATURES = [", 1)[1].split("];", 1)[0]
    flags = re.findall(r",\s*(true|false)\]", block)
    assert len(flags) == len(FEATURE_GROUPS["all"])
    assert flags.count("true") == len(FEATURE_GROUPS["geometric_h1"])


def test_warcraft_probe_control_matches_step_11() -> None:
    """The untrained-encoder probe reads terrain off the last hidden block about as well as the trained one.

    Step 11 reported R² 0.91 (trained) and 0.92 (untrained) at the last hidden
    block over all test maps; on single maps the two must stay close and high.
    """
    from scripts.demo import WarcraftState

    try:
        w = WarcraftState()
    except (FileNotFoundError, OSError) as error:
        pytest.skip(f"WarCraft data or model missing: {error}")
    for i in range(5):
        w.load_map(i)
        e = w.view["explain"]
        assert e["untrained_probe_r2"]["block3"] > 0.8 and e["probe_r2"]["block3"] > 0.8
        assert e["untrained_probe_r2"]["output"] < 0.5
    # repainting a tile with its own terrain is the seam control, and says so
    rc = (5, 5)
    w.paint(rc, float(w.weights[rc]))
    assert "Same terrain" in w.message and w.view["explain"] is not None
