"""Opportune sabotage: make the planner ignore a narrow shortcut without closing it.

The scenario: a map where the shortest route passes through a **gate**, a
3-cell opening in a wall, and every route that avoids the gate is much longer.
The attacker flips a few cells near the gate under two constraints:

1. the gate is never fully closed (at least `min_open` of its cells stay free);
2. the gate stays the right choice: on the altered map, the optimum *through*
   the gate is strictly shorter than the optimum with the gate closed.

Under (2), classical A* on the altered map still goes through the gate, so a
planner that takes the long way round is wrong by the project's own error
rule (deviation from the *altered* map's optimum). The sabotage succeeds when
Neural A* stops using the gate.

Why the scenarios are built rather than mined: the maze benchmark's openings
are as wide as a room (~10 cells), and `find_gates` finds no passage of four
cells or fewer on the optimal route of any test map whose closure costs a
single move. The generator (`make_scenario`) therefore draws maze-like maps —
thin walls, 32 x 32, the same 0/1 encoding — with one 3-cell gate and a long
detour, in random orientations. Which of them the planner solves correctly
before any attack is checked by `select_scenarios`, with the planner.

The numpy part (scenario construction, constraints, `find_gates`) needs no
torch; the planner part imports it lazily.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .data import MazeInstance
from .exact_solver import distance_field, solve_astar, solve_dijkstra
from .grid import RC, flip_cells, is_reachable

#: Side of the generated maps (the benchmark's).
MAP_SIZE: int = 32
#: Width of the gate, in cells.
GATE_WIDTH: int = 3
#: A scenario is kept only if closing its gate makes the optimum at least this
#: many times longer.
MIN_DETOUR_RATIO: float = 1.5
#: Generated scenarios get instance ids from here up, so they never collide
#: with the benchmark's (0-99) in caches keyed by instance id.
SCENARIO_ID_BASE: int = 1000
#: Default reach of the attack around the gate: the encoder's receptive-field
#: radius, so every candidate can move the cost of at least one gate cell.
DEFAULT_RADIUS: int = 5


# --------------------------------------------------------------------------- scenarios
@dataclass(frozen=True)
class GateScenario:
    """A planning problem whose shortest route goes through a narrow gate.

    Attributes:
        name: Short label, e.g. ``"wall-17"``.
        layout: ``"wall"`` (a long wall with a door) or ``"room"`` (the goal in
            a walled room with a door facing the start).
        seed: Seed that generated it.
        instance: The problem, with `instance_id` = ``SCENARIO_ID_BASE + seed``.
        gate: The gate's cells.
        base_cost: Optimum with the gate open.
        closed_cost: Optimum with the gate closed (the detour).
    """

    name: str
    layout: str
    seed: int
    instance: MazeInstance
    gate: Tuple[RC, ...]
    base_cost: float
    closed_cost: float

    @property
    def detour_ratio(self) -> float:
        """How many times longer the best route avoiding the gate is."""
        return self.closed_cost / self.base_cost


def _transform_rc(rc: RC, k: int, flip: bool, n: int) -> RC:
    """Cell after ``np.rot90(np.fliplr(a) if flip else a, k)`` on an n x n array."""
    r, c = int(rc[0]), int(rc[1])
    if flip:
        c = n - 1 - c
    for _ in range(k % 4):
        r, c = n - 1 - c, r
    return (r, c)


def _transform(m: np.ndarray, k: int, flip: bool) -> np.ndarray:
    return np.ascontiguousarray(np.rot90(np.fliplr(m) if flip else m, k))


def _wall_layout(rng: np.random.Generator, n: int):
    """A vertical wall across the map, a 3-cell door near the top, a wide opening at the bottom."""
    m = np.ones((n, n), dtype=np.float32)
    wc = int(rng.integers(12, 20))
    d = int(rng.integers(2, 8))
    a = int(rng.integers(4, 7))
    m[:, wc] = 0
    m[d : d + GATE_WIDTH, wc] = 1
    m[n - a :, wc] = 1
    gate = tuple((r, wc) for r in range(d, d + GATE_WIDTH))
    mid = d + 1
    start = (int(np.clip(mid + rng.integers(-2, 3), 0, n - 1)), int(rng.integers(1, wc - 4)))
    goal = (int(np.clip(mid + rng.integers(-2, 3), 0, n - 1)), int(rng.integers(wc + 5, n - 1)))
    # Maze-like partitions on both sides, each with a wide gap; they sit below
    # the door, so they lengthen the detour and leave the gate route alone.
    for lo, hi in ((0, wc), (wc + 1, n)):
        if hi - lo < 10:
            continue
        p = int(rng.integers(d + 7, n - a - 2))
        m[p, lo:hi] = 0
        g0 = int(rng.integers(lo, hi - 8 + 1))
        m[p, g0 : g0 + 8] = 1
    return m, gate, start, goal


def _room_layout(rng: np.random.Generator, n: int):
    """The goal inside a walled room: a 3-cell door facing the start, a back door far away."""
    m = np.ones((n, n), dtype=np.float32)
    s = int(rng.integers(12, 17))
    r0 = int(rng.integers(4, n - 2 - s))
    c0 = int(rng.integers(10, n - 2 - s))
    r1, c1 = r0 + s, c0 + s
    m[r0, c0 : c1 + 1] = 0
    m[r1, c0 : c1 + 1] = 0
    m[r0 : r1 + 1, c0] = 0
    m[r0 : r1 + 1, c1] = 0
    dd = int(rng.integers(r0 + 2, r1 - GATE_WIDTH - 1))
    m[dd : dd + GATE_WIDTH, c0] = 1
    gate = tuple((r, c0) for r in range(dd, dd + GATE_WIDTH))
    b = int(rng.integers(3, 6))
    bd = int(rng.integers(r0 + 1, r1 - b))
    m[bd : bd + b, c1] = 1  # the back door, on the far wall
    mid = dd + 1
    goal = (int(np.clip(mid + rng.integers(-3, 4), r0 + 1, r1 - 1)), int(rng.integers(c0 + 2, c0 + 6)))
    start = (int(np.clip(mid + rng.integers(-3, 4), 0, n - 1)), int(rng.integers(1, c0 - 3)))
    return m, gate, start, goal


def make_scenario(seed: int, layout: Optional[str] = None, n: int = MAP_SIZE,
                  min_detour_ratio: float = MIN_DETOUR_RATIO) -> Optional[GateScenario]:
    """Draw one gate scenario, or None if the draw fails the checks.

    The layout is drawn in a canonical orientation, then turned by a random
    quarter turn and optionally mirrored, so gates face every direction.

    Checks: start and goal are free, distinct and outside the gate; the goal is
    reachable with the gate open *and* closed; closing the gate makes the
    optimum at least `min_detour_ratio` times longer.

    Args:
        seed: Seed; the same seed always gives the same scenario.
        layout: ``"wall"``, ``"room"`` or None to draw one at random.
        n: Map side.
        min_detour_ratio: Minimum ``closed_cost / base_cost``.

    Returns:
        GateScenario | None: The scenario, or None when this seed's draw is
        rejected (callers just try the next seed).
    """
    rng = np.random.default_rng([seed, 7919])
    # Always drawn, so a seed gives the same map whether or not the layout is named.
    drawn = ("wall", "room")[int(rng.integers(2))]
    layout = layout or drawn
    builder = {"wall": _wall_layout, "room": _room_layout}[layout]
    m, gate, start, goal = builder(rng, n)
    k, flip = int(rng.integers(4)), bool(rng.integers(2))
    m = _transform(m, k, flip)
    gate = tuple(sorted(_transform_rc(rc, k, flip, n) for rc in gate))
    start, goal = _transform_rc(start, k, flip, n), _transform_rc(goal, k, flip, n)

    if start == goal or start in gate or goal in gate or m[start] == 0 or m[goal] == 0:
        return None
    base = solve_dijkstra(m, start, goal).cost
    closed = solve_dijkstra(close_gate(m, gate), start, goal).cost
    if not (np.isfinite(base) and np.isfinite(closed)) or closed < min_detour_ratio * base:
        return None
    instance = MazeInstance(
        instance_id=SCENARIO_ID_BASE + int(seed),
        split="gate",
        map_design=m,
        goal_rc=goal,
        start_rc=start,
        opt_dist=distance_field(m, goal),
        start_seed=int(seed),
    )
    return GateScenario(name=f"{layout}-{seed}", layout=layout, seed=int(seed), instance=instance,
                        gate=gate, base_cost=float(base), closed_cost=float(closed))


def scenario_candidates(seeds: Iterable[int], layout: Optional[str] = None) -> List[GateScenario]:
    """Every accepted scenario among `seeds`, in seed order."""
    out = []
    for seed in seeds:
        sc = make_scenario(int(seed), layout)
        if sc is not None:
            out.append(sc)
    return out


# --------------------------------------------------------------------------- constraints
def close_gate(map_design: np.ndarray, gate: Sequence[RC]) -> np.ndarray:
    """A copy of the map with every gate cell made an obstacle."""
    out = map_design.copy()
    for rc in gate:
        out[tuple(rc)] = 0
    return out


def open_gate_cells(map_design: np.ndarray, gate: Sequence[RC]) -> int:
    """How many gate cells are free."""
    return int(sum(map_design[tuple(rc)] == 1 for rc in gate))


def uses_gate(path_map: np.ndarray, gate: Sequence[RC]) -> bool:
    """Whether a binary route map steps on any gate cell.

    Crossing the gate's wall means landing on one of its cells (a move changes
    each coordinate by at most one), so a route through the gate always marks
    one of them.
    """
    return bool(any(path_map[tuple(rc)] > 0 for rc in gate))


def shortcut_status(map_design: np.ndarray, start: RC, goal: RC, gate: Sequence[RC]) -> Dict[str, float]:
    """Optimum on the map, optimum with the gate closed, and whether the gate is still the way.

    Returns:
        dict: ``optimal_cost``, ``closed_cost`` (``inf`` if closing the gate
        disconnects the goal), ``shortcut_optimal`` (every optimal route goes
        through the gate, i.e. ``optimal_cost < closed_cost``).
    """
    optimum = solve_dijkstra(map_design, start, goal).cost
    closed = solve_dijkstra(close_gate(map_design, gate), start, goal).cost
    return {"optimal_cost": float(optimum), "closed_cost": float(closed),
            "shortcut_optimal": bool(optimum < closed - 1e-9)}


def admissible(map_design: np.ndarray, start: RC, goal: RC, gate: Sequence[RC],
               min_open: int = 1) -> bool:
    """The attack's constraints: goal reachable, gate not closed, gate still the shortest way."""
    if open_gate_cells(map_design, gate) < min_open:
        return False
    if not is_reachable(map_design, start, goal):
        return False
    return shortcut_status(map_design, start, goal, gate)["shortcut_optimal"]


def neighbourhood(scenario: GateScenario, radius: int = DEFAULT_RADIUS) -> List[RC]:
    """Cells within Chebyshev distance `radius` of the gate, start and goal excluded, row-major."""
    inst = scenario.instance
    h, w = inst.map_design.shape
    rows = [rc[0] for rc in scenario.gate]
    cols = [rc[1] for rc in scenario.gate]
    cells = []
    for r in range(max(0, min(rows) - radius), min(h, max(rows) + radius + 1)):
        for c in range(max(0, min(cols) - radius), min(w, max(cols) + radius + 1)):
            if (r, c) not in (inst.start_rc, inst.goal_rc):
                cells.append((r, c))
    return cells


def find_gates(map_design: np.ndarray, start: RC, goal: RC, max_width: int = 4,
               min_extra: float = 1.0) -> List[Dict[str, object]]:
    """Narrow passages on the optimal route whose closure lengthens it (numpy only).

    For every cell of one optimal route, the free run through it along the row
    and along the column is taken; runs of at most `max_width` cells are
    candidate gates. A candidate is kept if closing it makes the optimum at
    least `min_extra` moves longer without disconnecting the goal.

    Used to check whether the benchmark has natural gates (it does not).

    Returns:
        list[dict]: ``gate``, ``width``, ``base_cost``, ``closed_cost``,
        ``extra``, sorted by ``extra``, largest first.
    """
    h, w = map_design.shape
    base = solve_dijkstra(map_design, start, goal)
    seen, out = set(), []
    for p in base.path[1:-1]:
        for dr, dc in ((0, 1), (1, 0)):
            run = [tuple(p)]
            for sign in (1, -1):
                r, c = p
                while True:
                    r, c = r + dr * sign, c + dc * sign
                    if not (0 <= r < h and 0 <= c < w) or map_design[r, c] == 0:
                        break
                    run.append((r, c))
                    if len(run) > max_width:
                        break
            gate = tuple(sorted(run))
            if len(gate) > max_width or gate in seen or start in gate or goal in gate:
                continue
            seen.add(gate)
            closed = solve_dijkstra(close_gate(map_design, gate), start, goal).cost
            if np.isfinite(closed) and closed - base.cost >= min_extra:
                out.append({"gate": gate, "width": len(gate), "base_cost": float(base.cost),
                            "closed_cost": float(closed), "extra": float(closed - base.cost)})
    return sorted(out, key=lambda g: -g["extra"])


# --------------------------------------------------------------------------- the planner
def plan_designs(planner, instance: MazeInstance, designs: Sequence[np.ndarray],
                 batch_size: int = 64) -> List[Dict[str, object]]:
    """Run Neural A* (encoder + search, both on the given map) on several maps of one problem.

    Returns:
        list[dict]: Per map: ``path_map``, ``histories``, ``cost_map`` (numpy
        ``(H, W)``), ``moves``, ``expanded``.
    """
    import torch

    from .metrics import n_expansions, path_cost
    from .planner import encode_cost_map, run_astar_module

    maps0, starts0, goals0 = instance.to_tensors()
    out_rows: List[Dict[str, object]] = []
    for begin in range(0, len(designs), batch_size):
        chunk = designs[begin : begin + batch_size]
        maps = torch.cat([torch.from_numpy(d.astype(np.float32)).reshape_as(maps0) for d in chunk])
        starts = starts0.expand(len(chunk), -1, -1, -1).contiguous()
        goals = goals0.expand(len(chunk), -1, -1, -1).contiguous()
        with torch.no_grad():
            costs = encode_cost_map(planner, maps, starts, goals)
            out = run_astar_module(planner, costs, starts, goals, maps)
        moves, expanded = path_cost(out.paths), n_expansions(out.histories)
        for i in range(len(chunk)):
            out_rows.append({
                "path_map": out.paths[i, 0].cpu().numpy(),
                "histories": out.histories[i, 0].cpu().numpy(),
                "cost_map": costs[i, 0].cpu().numpy(),
                "moves": float(moves[i]),
                "expanded": float(expanded[i]),
            })
    return out_rows


def assess(planner, scenario: GateScenario, flips: Sequence[RC] = (),
           planned: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """Everything the study reports about one altered version of a scenario.

    Args:
        planner: A `NeuralAstar` in eval mode.
        scenario: The scenario.
        flips: Cells flipped relative to the scenario's map.
        planned: A `plan_designs` entry for this map, if already computed.

    Returns:
        dict: ``flips``, ``n_flips``, ``moves``, ``optimal_cost``,
        ``closed_cost``, ``opt_ratio``, ``expanded``, ``classical_expanded``,
        ``planner_uses_gate``, ``shortcut_optimal``, ``gate_open``,
        ``success`` (the shortcut is still the way, the planner avoids it),
        and the maps ``design``, ``path_map``, ``histories``, ``cost_map``.
    """
    inst = scenario.instance
    design = flip_cells(inst.map_design, list(flips))
    planned = planned or plan_designs(planner, inst, [design])[0]
    status = shortcut_status(design, inst.start_rc, inst.goal_rc, scenario.gate)
    avoid = not uses_gate(planned["path_map"], scenario.gate)
    return {
        "flips": [tuple(int(v) for v in rc) for rc in flips],
        "n_flips": len(flips),
        "moves": planned["moves"],
        "optimal_cost": status["optimal_cost"],
        "closed_cost": status["closed_cost"],
        "opt_ratio": planned["moves"] / status["optimal_cost"],
        "expanded": planned["expanded"],
        "classical_expanded": float(solve_astar(design, inst.start_rc, inst.goal_rc).expansions),
        "planner_uses_gate": not avoid,
        "shortcut_optimal": status["shortcut_optimal"],
        "gate_open": open_gate_cells(design, scenario.gate),
        "success": bool(avoid and status["shortcut_optimal"]),
        "design": design,
        "path_map": planned["path_map"],
        "histories": planned["histories"],
        "cost_map": planned["cost_map"],
    }


def select_scenarios(planner, n: int = 3, seeds: Iterable[int] = range(400)) -> List[GateScenario]:
    """The first `n` scenarios on which the clean planner is optimal and uses the gate.

    A sabotage only means something where the untouched planner was right, as
    in the rest of the project. Seeds are tried in order; no layout may take
    more than ``ceil(n / 2)`` places in a first pass, so the selection mixes
    walls and rooms when it can, and a second pass fills any places left.

    Args:
        planner: A `NeuralAstar` in eval mode.
        n: How many to return.
        seeds: Seeds to try, in order.

    Returns:
        list[GateScenario]: Up to `n` scenarios (fewer if the seeds run out).
    """
    usable: List[GateScenario] = []
    for sc in scenario_candidates(seeds):
        clean = assess(planner, sc)
        if clean["planner_uses_gate"] and clean["moves"] <= clean["optimal_cost"] + 1e-6:
            usable.append(sc)
            cap = -(-n // 2)
            if len(_mixed(usable, n, cap)) >= n:
                break
    chosen = _mixed(usable, n, -(-n // 2))
    for sc in usable:
        if len(chosen) >= n:
            break
        if all(sc is not c for c in chosen):
            chosen.append(sc)
    return chosen


def _mixed(usable: Sequence[GateScenario], n: int, cap: int) -> List[GateScenario]:
    out: List[GateScenario] = []
    for sc in usable:
        if len(out) < n and sum(o.layout == sc.layout for o in out) < cap:
            out.append(sc)
    return out


@dataclass
class SabotageResult:
    """The outcome of one attack.

    Attributes:
        scenario: The attacked scenario.
        rounds: `assess` dicts (without the maps), the clean map first, then
            one per greedy round.
        flips: The final flip set.
        success: Whether the final map fools the planner.
        evaluated: Number of altered maps the attack ran the planner on.
    """

    scenario: GateScenario
    rounds: List[Dict[str, object]] = field(default_factory=list)
    flips: List[RC] = field(default_factory=list)
    success: bool = False
    evaluated: int = 0


def _strip(row: Dict[str, object]) -> Dict[str, object]:
    return {k: v for k, v in row.items() if k not in ("design", "path_map", "histories", "cost_map")}


def sabotage_attack(planner, scenario: GateScenario, budget: int = 4, radius: int = DEFAULT_RADIUS,
                    min_open: int = 1, stop_on_success: bool = True,
                    initial: Sequence[RC] = ()) -> SabotageResult:
    """Greedy constrained attack: add the flip that hurts the route most, round by round.

    Every round tries each cell of the gate's neighbourhood (`neighbourhood`)
    as an addition to the current flip set, discards additions that break the
    constraints (`admissible`: gate never fully closed, gate still the
    shortest way, goal reachable), runs the planner on the rest in one batch,
    and keeps the best by ``(success, opt_ratio, expanded)``. With
    `stop_on_success` it stops at the first round that fools the planner, so
    the flip count is as small as greedy search finds.

    Args:
        planner: A `NeuralAstar` in eval mode.
        scenario: The scenario to attack.
        budget: Maximum number of flips.
        radius: Neighbourhood radius around the gate.
        min_open: Gate cells that must stay free.
        stop_on_success: Stop once the planner avoids the gate.
        initial: Flips already applied (the demo adds one round per click).

    Returns:
        SabotageResult: Rounds, final flips and verdict.
    """
    inst = scenario.instance
    chosen: List[RC] = [tuple(int(v) for v in rc) for rc in initial]
    result = SabotageResult(scenario=scenario)
    current = assess(planner, scenario, chosen)
    result.rounds.append(_strip(current))
    pool = neighbourhood(scenario, radius)
    for _ in range(budget - len(chosen)):
        if stop_on_success and current["success"]:
            break
        sets, designs = [], []
        for cand in pool:
            if cand in chosen:
                continue
            cells = chosen + [cand]
            design = flip_cells(inst.map_design, cells)
            if admissible(design, inst.start_rc, inst.goal_rc, scenario.gate, min_open):
                sets.append(cells)
                designs.append(design)
        if not designs:
            break
        planned = plan_designs(planner, inst, designs)
        result.evaluated += len(designs)
        best_i = max(range(len(designs)), key=lambda i: (
            not uses_gate(planned[i]["path_map"], scenario.gate),
            planned[i]["moves"],
            planned[i]["expanded"],
        ))
        chosen = sets[best_i]
        current = assess(planner, scenario, chosen, planned[best_i])
        result.rounds.append(_strip(current))
    result.flips = chosen
    result.success = bool(current["success"])
    return result


def random_sabotage(planner, scenario: GateScenario, budget: int, draws: int = 200,
                    radius: int = DEFAULT_RADIUS, min_open: int = 1, seed: int = 0,
                    max_attempts: int = 50) -> Dict[str, float]:
    """The control: `budget` random admissible flips in the same neighbourhood.

    Same constraints and same cells as `sabotage_attack`, no information about
    the planner. The gap between the two is what targeting buys.

    Returns:
        dict: ``draws`` (admissible draws found), ``success_rate``,
        ``mean_opt_ratio``, ``gate_avoided_rate``.
    """
    inst = scenario.instance
    rng = np.random.default_rng([seed, scenario.seed, budget])
    pool = neighbourhood(scenario, radius)
    designs = []
    for _ in range(draws):
        for _ in range(max_attempts):
            idx = rng.choice(len(pool), size=budget, replace=False)
            design = flip_cells(inst.map_design, [pool[i] for i in idx])
            if admissible(design, inst.start_rc, inst.goal_rc, scenario.gate, min_open):
                designs.append(design)
                break
    if not designs:
        return {"draws": 0, "success_rate": float("nan"), "mean_opt_ratio": float("nan"),
                "gate_avoided_rate": float("nan")}
    planned = plan_designs(planner, inst, designs)
    avoided = np.array([not uses_gate(p["path_map"], scenario.gate) for p in planned])
    ratios = np.array([p["moves"] / solve_dijkstra(d, inst.start_rc, inst.goal_rc).cost
                       for p, d in zip(planned, designs)])
    # Every draw is admissible, so the shortcut is optimal on all of them:
    # avoiding the gate is a success.
    return {"draws": len(designs), "success_rate": float(avoided.mean()),
            "mean_opt_ratio": float(ratios.mean()), "gate_avoided_rate": float(avoided.mean())}
