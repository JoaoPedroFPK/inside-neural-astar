"""Opportune sabotage (notebook 03): scenarios are what they claim, the attack keeps its constraints."""

from __future__ import annotations

import numpy as np

from src.exact_solver import solve_dijkstra
from src.grid import flip_cells
from src.sabotage import (
    MIN_DETOUR_RATIO,
    admissible,
    close_gate,
    find_gates,
    make_scenario,
    neighbourhood,
    sabotage_attack,
    scenario_candidates,
    select_scenarios,
    shortcut_status,
    uses_gate,
)
from src.symmetry import path_to_map


def test_scenarios_are_deterministic_and_have_a_real_shortcut() -> None:
    """Same seed, same map; the optimal route uses the gate; closing it costs a long detour."""
    pool = scenario_candidates(range(40))
    assert len(pool) >= 20 and {sc.layout for sc in pool} == {"wall", "room"}
    for sc in pool:
        inst = sc.instance
        again = make_scenario(sc.seed, sc.layout)
        assert np.array_equal(again.instance.map_design, inst.map_design) and again.gate == sc.gate
        assert all(inst.map_design[rc] == 1 for rc in sc.gate) and len(sc.gate) == 3
        path = solve_dijkstra(inst.map_design, inst.start_rc, inst.goal_rc).path
        assert uses_gate(path_to_map(path, inst.map_design.shape), sc.gate)
        assert sc.closed_cost >= MIN_DETOUR_RATIO * sc.base_cost
        assert solve_dijkstra(close_gate(inst.map_design, sc.gate), inst.start_rc, inst.goal_rc).cost == sc.closed_cost


def test_constraints_allow_narrowing_but_not_closing() -> None:
    sc = scenario_candidates(range(10))[0]
    inst, gate = sc.instance, sc.gate
    args = (inst.start_rc, inst.goal_rc, gate)
    assert admissible(inst.map_design, *args)
    assert admissible(flip_cells(inst.map_design, gate[:2]), *args)
    assert not admissible(flip_cells(inst.map_design, gate), *args)
    assert not admissible(flip_cells(inst.map_design, gate[:1]), *args, min_open=3)
    assert shortcut_status(inst.map_design, *args)["shortcut_optimal"]


def test_neighbourhood_stays_near_the_gate_and_skips_endpoints() -> None:
    sc = scenario_candidates(range(10))[1]
    cells = neighbourhood(sc, radius=5)
    assert sc.instance.start_rc not in cells and sc.instance.goal_rc not in cells
    assert all(min(max(abs(r - g[0]), abs(c - g[1])) for g in sc.gate) <= 5 for r, c in cells)
    assert all(g in cells for g in sc.gate)


def test_find_gates_sees_a_built_gate_and_none_in_the_benchmark(test_split) -> None:
    """A built 3-cell gate is found; the test mazes have no natural one (their openings are room-wide)."""
    sc = scenario_candidates(range(10))[0]
    inst = sc.instance
    found = find_gates(inst.map_design, inst.start_rc, inst.goal_rc, max_width=4)
    assert any(set(g["gate"]) == set(sc.gate) for g in found)
    for i in range(len(test_split)):
        t = test_split.instance(i)
        assert not find_gates(t.map_design, t.start_rc, t.goal_rc, max_width=4)


# ------------------------------------------------------------------ with the planner
def test_attack_never_breaks_its_constraints(pretrained_planner) -> None:
    """Every round leaves a gate cell open and the gate strictly shortest."""
    scenarios = select_scenarios(pretrained_planner, n=2, seeds=range(200))
    assert scenarios, "the clean planner solves no shortcut map"
    for sc in scenarios:
        result = sabotage_attack(pretrained_planner, sc, budget=3, stop_on_success=False)
        assert result.rounds[0]["planner_uses_gate"] and not result.rounds[0]["success"]
        for row in result.rounds[1:]:
            assert row["gate_open"] >= 1 and row["shortcut_optimal"]
            design = flip_cells(sc.instance.map_design, row["flips"])
            assert admissible(design, sc.instance.start_rc, sc.instance.goal_rc, sc.gate)
            assert row["success"] == (not row["planner_uses_gate"])
