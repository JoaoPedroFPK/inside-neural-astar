"""Map alterations (cell flips) and their evaluation — E2.

A **flip** inverts one cell's polarity: a free cell becomes an obstacle or an
obstacle becomes free. An **alteration** is a set of `budget` flips applied to
one instance, subject to the reachability constraint (the goal must stay
reachable from the fixed start).

This module provides the sampler for the random baseline (PLAN.md step 3) and
the evaluation harness every later step reuses. Only the *random* method lives
here so far; `exhaustive` (step 6), `greedy` (step 9) and `gradient` (step 9,
conditional) get added when those steps come up.

What an evaluation measures, for each alteration:

* ``C0O0`` — the planner on the original map, the baseline. Recomputed per
  instance and cached, so "did this flip break it?" can be separated from "was
  it already suboptimal?" (17 of 100 test instances are).
* ``C1O1`` — the planner on the altered map: the encoder sees the altered map
  *and* the search uses the altered obstacle mask. This is the full alteration,
  the condition a deployed planner would actually be in.
* The reference on the altered map: the certified optimum
  (`src.exact_solver.solve_dijkstra`) and classical A\\*'s expansion count
  (`src.exact_solver.solve_astar`). Both come from the **altered** map, because
  classical A\\* replanning after a flip is the correct response, not an error.

The two swap conditions ``C0O1`` and ``C1O0`` are PLAN.md step 7 and are not
computed here; the row schema leaves room for them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple  # noqa: F401

import numpy as np
import torch

from .data import MazeInstance
from .exact_solver import solve_astar, solve_dijkstra
from .grid import RC, flip_cells, is_reachable, validate_path_map
from .metrics import exp_ratio, n_expansions, opt_ratio, path_cost
from .planner import encode_cost_map, run_astar_module

#: Default cap on resampling attempts before a draw is declared impossible.
MAX_SAMPLE_ATTEMPTS: int = 200


def candidate_cells(
    map_design: np.ndarray,
    start_rc: RC,
    goal_rc: RC,
    free_only: bool = False,
) -> List[RC]:
    """Cells eligible to be flipped, in row-major order.

    Start and goal are always excluded: flipping either changes the problem
    instead of the map, and the metrics would no longer be comparable.

    Args:
        map_design: ``(H, W)`` binary map, ``1`` = free.
        start_rc: Fixed start cell.
        goal_rc: Goal cell.
        free_only: If True, only free cells (so flips can only *add*
            obstacles). The default False matches PLAN.md step 6, which sweeps
            "each candidate cell (~1,000)", i.e. the whole grid.

    Returns:
        list[tuple[int, int]]: Eligible cells. With `free_only=False` this is
        every cell but two, so ``H * W - 2`` of them.
    """
    cells = []
    for r in range(map_design.shape[0]):
        for c in range(map_design.shape[1]):
            rc = (r, c)
            if rc == start_rc or rc == goal_rc:
                continue
            if free_only and map_design[rc] != 1:
                continue
            cells.append(rc)
    return cells


def sample_random_flips(
    instance: MazeInstance,
    budget: int,
    rng: np.random.Generator,
    free_only: bool = False,
    max_attempts: int = MAX_SAMPLE_ATTEMPTS,
) -> Optional[Tuple[RC, ...]]:
    """Draw `budget` distinct cells whose flip keeps the goal reachable.

    Rejection sampling: draw a set, test reachability, redraw on failure. The
    whole set is redrawn rather than repaired one cell at a time, so the result
    stays a uniform draw from the admissible sets of that size.

    Args:
        instance: The problem instance (supplies the map, start and goal).
        budget: Number of cells to flip, ``k`` in PLAN.md.
        rng: Seeded generator; the only source of randomness.
        free_only: Restrict candidates to free cells (obstacle-adding flips).
        max_attempts: Give up after this many rejected draws.

    Returns:
        tuple[tuple[int, int], ...] | None: The flipped cells in row-major
        order, or ``None`` if no admissible draw was found within
        `max_attempts` (e.g. a corridor instance where almost any wall
        disconnects the goal).

    Raises:
        ValueError: If `budget` is not positive or exceeds the candidate count.
    """
    if budget <= 0:
        raise ValueError(f"budget must be positive, got {budget}")
    cells = candidate_cells(instance.map_design, instance.start_rc, instance.goal_rc, free_only)
    if budget > len(cells):
        raise ValueError(f"budget {budget} exceeds {len(cells)} candidate cells")

    for _ in range(max_attempts):
        chosen = [cells[i] for i in rng.choice(len(cells), size=budget, replace=False)]
        altered = flip_cells(instance.map_design, chosen)
        if is_reachable(altered, instance.start_rc, instance.goal_rc):
            return tuple(sorted(chosen))
    return None


@dataclass
class Baseline:
    """Planner and reference numbers on an **unaltered** instance (``C0O0``).

    Attributes:
        cost_map: ``(H, W)`` encoder output on the original map.
        path_cost: Planner path cost (number of moves).
        expansions: Planner expansion count.
        optimal_cost: Certified optimum on the original map.
        classical_expansions: Classical A\\* expansions on the original map.
        opt_ratio: ``path_cost / optimal_cost``.
        exp_ratio: ``expansions / classical_expansions``.
    """

    cost_map: np.ndarray
    path_cost: float
    expansions: float
    optimal_cost: float
    classical_expansions: float
    opt_ratio: float
    exp_ratio: float


def compute_baseline(planner, instance: MazeInstance) -> Baseline:
    """Evaluate the planner and the solver on an instance's original map.

    Args:
        planner: A `NeuralAstar` planner in eval mode.
        instance: The problem instance.

    Returns:
        Baseline: The ``C0O0`` numbers, including the original cost map (needed
        for the ΔCost columns of the alteration rows).
    """
    maps, starts, goals = instance.to_tensors()
    with torch.no_grad():
        cost_map = encode_cost_map(planner, maps, starts, goals)
        out = run_astar_module(planner, cost_map, starts, goals, maps)

    classical = solve_astar(instance.map_design, instance.start_rc, instance.goal_rc)
    optimum = solve_dijkstra(instance.map_design, instance.start_rc, instance.goal_rc)
    cost = float(path_cost(out.paths)[0])
    expansions = float(n_expansions(out.histories)[0])
    return Baseline(
        cost_map=cost_map[0, 0].numpy(),
        path_cost=cost,
        expansions=expansions,
        optimal_cost=float(optimum.cost),
        classical_expansions=float(classical.expansions),
        opt_ratio=float(opt_ratio(np.array([cost]), np.array([optimum.cost]))[0]),
        exp_ratio=float(exp_ratio(np.array([expansions]), np.array([classical.expansions]))[0]),
    )


@dataclass
class AlterationRequest:
    """One alteration to evaluate.

    Attributes:
        instance: The instance to alter.
        flip_cells: Cells to flip.
        method: How the cells were chosen (``random`` | ``exhaustive`` |
            ``greedy`` | ``gradient``), recorded in the results row.
        objective: What the choice optimised (``optimality`` | ``efficiency`` |
            ``none``); ``none`` for the random baseline.
        draw: Index of the random draw, for reproducibility bookkeeping.
    """

    instance: MazeInstance
    flip_cells: Tuple[RC, ...]
    method: str = "random"
    objective: str = "none"
    draw: int = 0
    extra: Dict[str, object] = field(default_factory=dict)


def evaluate_alterations(
    planner,
    requests: Sequence[AlterationRequest],
    baselines: Optional[Dict[int, Baseline]] = None,
    batch_size: int = 64,
    seed: Optional[int] = None,
) -> List[Dict[str, object]]:
    """Evaluate alterations and return one results row each.

    The planner is run in batches (the encoder is the expensive part), while the
    exact solver runs per alteration — it is pure numpy and costs ~1 ms.

    Args:
        planner: A `NeuralAstar` planner in eval mode.
        requests: The alterations to evaluate.
        baselines: Optional cache of `Baseline` per ``instance_id``; missing
            entries are computed and inserted, so passing a dict across calls
            avoids recomputing ``C0O0``.
        batch_size: Number of altered maps per planner forward pass.
        seed: Recorded in each row (the start-selection seed of the study), not
            used for anything stochastic — this function is deterministic.

    Returns:
        list[dict]: One row per request, with the columns of the results
        manifest that apply to a full alteration: identifiers, ``opt_C0O0``,
        ``opt_C1O1``, ``exp_C0O0``, ``exp_C1O1``, the altered map's reference
        numbers, and the ΔCost columns. The swap columns ``opt_C0O1`` /
        ``opt_C1O0`` / ``exp_C0O1`` / ``exp_C1O0`` are left out; PLAN.md step 7
        adds them.

    Raises:
        ValueError: If a request's flips disconnect the goal (callers must
            filter with `sample_random_flips`, which enforces it).
    """
    baselines = {} if baselines is None else baselines
    rows: List[Dict[str, object]] = []

    for begin in range(0, len(requests), batch_size):
        chunk = requests[begin : begin + batch_size]
        altered_maps, starts, goals = [], [], []
        for request in chunk:
            instance = request.instance
            altered = flip_cells(instance.map_design, request.flip_cells)
            if not is_reachable(altered, instance.start_rc, instance.goal_rc):
                raise ValueError(
                    f"instance {instance.instance_id}: flips {request.flip_cells} "
                    "disconnect the goal; filter these out before evaluating"
                )
            maps_t, starts_t, goals_t = instance.to_tensors()
            altered_maps.append(torch.from_numpy(altered).reshape_as(maps_t))
            starts.append(starts_t)
            goals.append(goals_t)

        maps_batch = torch.cat(altered_maps)
        starts_batch = torch.cat(starts)
        goals_batch = torch.cat(goals)
        with torch.no_grad():
            cost_maps = encode_cost_map(planner, maps_batch, starts_batch, goals_batch)
            out = run_astar_module(planner, cost_maps, starts_batch, goals_batch, maps_batch)
        costs = path_cost(out.paths)
        expansions = n_expansions(out.histories)

        for k, request in enumerate(chunk):
            instance = request.instance
            if instance.instance_id not in baselines:
                baselines[instance.instance_id] = compute_baseline(planner, instance)
            baseline = baselines[instance.instance_id]

            altered = flip_cells(instance.map_design, request.flip_cells)
            optimum = solve_dijkstra(altered, instance.start_rc, instance.goal_rc)
            classical = solve_astar(altered, instance.start_rc, instance.goal_rc)

            altered_cost_map = cost_maps[k, 0].numpy()
            dcost = np.abs(altered_cost_map - baseline.cost_map)
            dcost_l1 = float(dcost.sum())
            under_flips = float(sum(dcost[rc] for rc in request.flip_cells))

            path_ok, path_reason = validate_path_map(
                out.paths[k, 0].numpy(), altered, instance.start_rc, instance.goal_rc
            )
            rows.append(
                {
                    "instance_id": instance.instance_id,
                    "split": instance.split,
                    "seed": instance.start_seed if seed is None else seed,
                    "start_rc": instance.start_rc,
                    "goal_rc": instance.goal_rc,
                    "budget": len(request.flip_cells),
                    "flip_cells": list(request.flip_cells),
                    "method": request.method,
                    "objective": request.objective,
                    "draw": request.draw,
                    "reachable": True,
                    "opt_C0O0": baseline.opt_ratio,
                    "opt_C1O1": float(opt_ratio(costs[k : k + 1], np.array([optimum.cost]))[0]),
                    "exp_C0O0": baseline.exp_ratio,
                    "exp_C1O1": float(
                        exp_ratio(expansions[k : k + 1], np.array([classical.expansions]))[0]
                    ),
                    "path_cost_C0O0": baseline.path_cost,
                    "path_cost_C1O1": float(costs[k]),
                    "expansions_C0O0": baseline.expansions,
                    "expansions_C1O1": float(expansions[k]),
                    "classical_opt_cost_original": baseline.optimal_cost,
                    "classical_expansions_original": baseline.classical_expansions,
                    "classical_opt_cost_altered": float(optimum.cost),
                    "classical_expansions_altered": float(classical.expansions),
                    "dcost_l1": dcost_l1,
                    "dcost_frac_under_flips": (under_flips / dcost_l1) if dcost_l1 > 0 else np.nan,
                    "n_obstacles_added": int(
                        sum(1 for rc in request.flip_cells if instance.map_design[rc] == 1)
                    ),
                    "n_obstacles_removed": int(
                        sum(1 for rc in request.flip_cells if instance.map_design[rc] == 0)
                    ),
                    "path_valid_C1O1": bool(path_ok),
                    "path_invalid_reason": "" if path_ok else path_reason,
                    **request.extra,
                }
            )
    return rows


def summarize_rows(
    rows: Iterable[Dict[str, object]],
    exp_threshold: float = 1.5,
) -> List[Dict[str, object]]:
    """Aggregate alteration rows into the per-budget fragility table of step 3.

    Args:
        rows: Rows from `evaluate_alterations`.
        exp_threshold: The ``Exp`` ratio above which an alteration counts as an
            efficiency failure. PLAN.md step 3 uses 1.5.

    Returns:
        list[dict]: One entry per budget, sorted by budget, with

        * ``frac_opt_deviating`` — share of alterations whose ``Opt`` exceeds
          1.0 on the altered map (PLAN.md's headline number);
        * ``frac_newly_suboptimal`` — the same share restricted to alterations
          whose baseline was already optimal, which is the fragility *caused*
          by the flip;
        * ``frac_exp_above_threshold`` and ``frac_newly_exp_above_threshold``,
          analogously for efficiency;
        * mean ratios, and the count of alterations evaluated.
    """
    by_budget: Dict[int, List[Dict[str, object]]] = {}
    for row in rows:
        by_budget.setdefault(int(row["budget"]), []).append(row)

    table = []
    for budget in sorted(by_budget):
        group = by_budget[budget]
        opt_altered = np.array([r["opt_C1O1"] for r in group], dtype=np.float64)
        opt_base = np.array([r["opt_C0O0"] for r in group], dtype=np.float64)
        exp_altered = np.array([r["exp_C1O1"] for r in group], dtype=np.float64)
        exp_base = np.array([r["exp_C0O0"] for r in group], dtype=np.float64)

        was_optimal = opt_base <= 1.0
        was_efficient = exp_base <= exp_threshold
        opt_original = np.array(
            [r["classical_opt_cost_original"] for r in group], dtype=np.float64
        )
        opt_altered_map = np.array(
            [r["classical_opt_cost_altered"] for r in group], dtype=np.float64
        )
        newly_broken = opt_altered[was_optimal][opt_altered[was_optimal] > 1.0]
        table.append(
            {
                "budget": budget,
                "n": len(group),
                "frac_optimum_changed": float(np.mean(opt_altered_map != opt_original)),
                "frac_optimum_shorter": float(np.mean(opt_altered_map < opt_original)),
                "mean_opt_when_newly_broken": (
                    float(newly_broken.mean()) if newly_broken.size else np.nan
                ),
                "max_opt_when_newly_broken": (
                    float(newly_broken.max()) if newly_broken.size else np.nan
                ),
                "frac_opt_deviating": float(np.mean(opt_altered > 1.0)),
                "frac_newly_suboptimal": (
                    float(np.mean(opt_altered[was_optimal] > 1.0)) if was_optimal.any() else np.nan
                ),
                "frac_exp_above_threshold": float(np.mean(exp_altered > exp_threshold)),
                "frac_newly_exp_above_threshold": (
                    float(np.mean(exp_altered[was_efficient] > exp_threshold))
                    if was_efficient.any()
                    else np.nan
                ),
                "mean_opt_C0O0": float(np.mean(opt_base)),
                "mean_opt_C1O1": float(np.mean(opt_altered)),
                "mean_exp_C0O0": float(np.mean(exp_base)),
                "mean_exp_C1O1": float(np.mean(exp_altered)),
                "mean_dcost_l1": float(np.mean([r["dcost_l1"] for r in group])),
                "mean_dcost_frac_under_flips": float(
                    np.nanmean([r["dcost_frac_under_flips"] for r in group])
                ),
            }
        )
    return table


def enumerate_k1_flips(
    instance: MazeInstance,
    free_only: bool = False,
) -> List[Tuple[RC, ...]]:
    """All single-cell flips of an instance that keep the goal reachable.

    The candidate set of PLAN.md step 6 ("each candidate cell, ~1,000"), with
    the reachability constraint already applied so callers never have to filter.
    Used by the step-3 worst-case probe and by step 6 itself.

    Args:
        instance: The problem instance.
        free_only: Restrict to free cells (obstacle-adding flips only).

    Returns:
        list[tuple[tuple[int, int]]]: One single-element tuple per admissible
        cell, in row-major order. Inadmissible cells (those whose flip
        disconnects the goal) are omitted; the count is reported by callers as
        the rejection rate.
    """
    admissible = []
    for rc in candidate_cells(instance.map_design, instance.start_rc, instance.goal_rc, free_only):
        altered = flip_cells(instance.map_design, [rc])
        if is_reachable(altered, instance.start_rc, instance.goal_rc):
            admissible.append((rc,))
    return admissible


def summarize_by_direction(
    rows: Iterable[Dict[str, object]], budget: int = 1
) -> List[Dict[str, object]]:
    """Split one budget's alterations by what the flip did to the map.

    On these maps the two directions are qualitatively different edits (see
    `docs/verified.md` § 11): the walls are one cell thick and the open areas
    are ~10 cells wide, so flipping a free cell drops an isolated **pillar**
    into open space, while flipping an obstacle opens a **door** between two
    areas. Reporting them together hides a large difference in damage rate.

    Args:
        rows: Rows from `evaluate_alterations`.
        budget: Which budget to split. Only single-direction budgets are
            meaningful, so the default is k=1; larger budgets mix directions
            within one alteration and are reported by `summarize_rows`.

    Returns:
        list[dict]: One entry per direction (``"free->obstacle (pillar)"`` /
        ``"obstacle->free (door)"``) with the share of draws, the
        newly-suboptimal rate and the mean ΔCost L1.
    """
    groups: Dict[str, List[Dict[str, object]]] = {}
    selected = [r for r in rows if int(r["budget"]) == budget]
    for row in selected:
        if int(row["n_obstacles_removed"]) == 0:
            label = "free->obstacle (pillar)"
        elif int(row["n_obstacles_added"]) == 0:
            label = "obstacle->free (door)"
        else:
            label = "mixed"
        groups.setdefault(label, []).append(row)

    out = []
    for label in sorted(groups):
        group = groups[label]
        opt_base = np.array([r["opt_C0O0"] for r in group], dtype=np.float64)
        opt_altered = np.array([r["opt_C1O1"] for r in group], dtype=np.float64)
        was_optimal = opt_base <= 1.0
        out.append(
            {
                "direction": label,
                "n": len(group),
                "share_of_draws": len(group) / max(len(selected), 1),
                "frac_newly_suboptimal": (
                    float(np.mean(opt_altered[was_optimal] > 1.0)) if was_optimal.any() else np.nan
                ),
                "mean_opt_C1O1": float(np.mean(opt_altered)),
                "mean_dcost_l1": float(np.mean([r["dcost_l1"] for r in group])),
            }
        )
    return out


#: Cap on the Opt ratio an alteration may cost while still counting as an
#: *efficiency* attack (plan step 6: "max Exp-ratio with Opt-ratio <= 1.05").
#: Without it the efficiency objective would simply select flips that wreck the
#: path, since a wrecked path usually also expands more nodes.
OPT_CAP_FOR_EFFICIENCY: float = 1.05


def select_worst_flips(
    rows: Sequence[Dict[str, object]],
    opt_cap: float = OPT_CAP_FOR_EFFICIENCY,
) -> List[Dict[str, object]]:
    """Tag, per instance, the worst alteration under each objective.

    PLAN.md step 6 selects two alterations per map: the one that damages path
    quality most, and the one that damages search efficiency most *without*
    materially damaging path quality. The second constraint is what keeps the
    two objectives distinct.

    Args:
        rows: Alteration rows from `evaluate_alterations`, any number of
            instances and budgets mixed together. Rows are grouped by
            ``(instance_id, budget)``.
        opt_cap: Maximum ``opt_C1O1`` an alteration may have to qualify for the
            efficiency objective.

    Returns:
        list[dict]: The same row objects, with two keys added in place and
        returned for convenience:

        * ``is_worst_optimality`` — True for the row maximising ``opt_C1O1``
          within its ``(instance_id, budget)`` group;
        * ``is_worst_efficiency`` — True for the row maximising ``exp_C1O1``
          among rows with ``opt_C1O1 <= opt_cap``.

        Ties are broken by the first row in iteration order, which is row-major
        by flipped cell, so the choice is deterministic. A group where no row
        satisfies the cap gets no efficiency winner.
    """
    groups: Dict[Tuple[int, int], List[Dict[str, object]]] = {}
    for row in rows:
        row["is_worst_optimality"] = False
        row["is_worst_efficiency"] = False
        groups.setdefault((int(row["instance_id"]), int(row["budget"])), []).append(row)

    for group in groups.values():
        best_opt = max(group, key=lambda r: float(r["opt_C1O1"]))
        best_opt["is_worst_optimality"] = True
        eligible = [r for r in group if float(r["opt_C1O1"]) <= opt_cap]
        if eligible:
            best_exp = max(eligible, key=lambda r: float(r["exp_C1O1"]))
            best_exp["is_worst_efficiency"] = True
    return list(rows)


def summarize_sweep(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Per-instance summary of an exhaustive sweep.

    Args:
        rows: Rows from `evaluate_alterations`, already tagged by
            `select_worst_flips`.

    Returns:
        dict: ``per_instance`` (one entry per instance with the baseline, the
        worst value under each objective and the share of damaging cells) and
        ``overall`` aggregates across instances.
    """
    by_instance: Dict[int, List[Dict[str, object]]] = {}
    for row in rows:
        by_instance.setdefault(int(row["instance_id"]), []).append(row)

    per_instance = []
    for instance_id in sorted(by_instance):
        group = by_instance[instance_id]
        opt = np.array([float(r["opt_C1O1"]) for r in group])
        exp = np.array([float(r["exp_C1O1"]) for r in group])
        baseline_opt = float(group[0]["opt_C0O0"])
        baseline_exp = float(group[0]["exp_C0O0"])
        eligible = opt <= OPT_CAP_FOR_EFFICIENCY
        worst_opt_row = next(r for r in group if r.get("is_worst_optimality"))
        worst_exp_row = next((r for r in group if r.get("is_worst_efficiency")), None)
        per_instance.append(
            {
                "instance_id": instance_id,
                "n_alterations": len(group),
                "baseline_opt": baseline_opt,
                "baseline_exp": baseline_exp,
                "worst_opt": float(opt.max()),
                "worst_opt_cell": list(worst_opt_row["flip_cells"][0]),
                "worst_exp_within_cap": float(exp[eligible].max()) if eligible.any() else np.nan,
                "worst_exp_cell": (
                    list(worst_exp_row["flip_cells"][0]) if worst_exp_row else None
                ),
                "frac_cells_damaging_opt": float(np.mean(opt > max(baseline_opt, 1.0))),
                "frac_cells_exp_above_1_5": float(np.mean(exp > 1.5)),
                "mean_opt": float(opt.mean()),
                "mean_exp": float(exp.mean()),
            }
        )

    worst_opt_values = np.array([p["worst_opt"] for p in per_instance])
    baselines = np.array([p["baseline_opt"] for p in per_instance])
    damaging = np.array([p["frac_cells_damaging_opt"] for p in per_instance])
    optimal_at_baseline = baselines <= 1.0
    return {
        "per_instance": per_instance,
        "overall": {
            "n_instances": len(per_instance),
            "n_alterations": int(sum(p["n_alterations"] for p in per_instance)),
            "frac_instances_with_a_damaging_cell": float(
                np.mean(worst_opt_values[optimal_at_baseline] > 1.0)
                if optimal_at_baseline.any()
                else np.nan
            ),
            "n_instances_optimal_at_baseline": int(optimal_at_baseline.sum()),
            "median_frac_cells_damaging": float(np.median(damaging)),
            "mean_frac_cells_damaging": float(np.mean(damaging)),
            "max_worst_opt": float(worst_opt_values.max()),
            "median_worst_opt": float(np.median(worst_opt_values[optimal_at_baseline]))
            if optimal_at_baseline.any()
            else np.nan,
        },
    }


def greedy_flip_search(
    planner,
    instance: MazeInstance,
    budget: int,
    objective: str,
    candidate_pool: Sequence[RC],
    baselines: Optional[Dict[int, Baseline]] = None,
    batch_size: int = 128,
    opt_cap: float = OPT_CAP_FOR_EFFICIENCY,
) -> List[Dict[str, object]]:
    """Grow a damaging flip set one cell at a time, keeping the best each round.

    Exhaustive search is only affordable at k=1: the number of cell *sets*
    grows as ``C(1022, k)``. Greedy search replaces it with ``budget`` rounds of
    single-cell choices, each conditioned on the cells chosen so far. It is an
    approximation — a pair of cells that only hurts jointly will be missed —
    and the random baseline at the same budgets is what keeps that bias
    visible.

    The candidate pool is supplied by the caller rather than being the whole
    grid, because the exhaustive k=1 sweep already ranks every cell: passing
    its top cells turns an 8 x 1022 search into 8 x |pool|. That prunes
    combinations whose first cell is individually harmless, which is the
    second approximation and is logged with the results.

    Args:
        planner: A `NeuralAstar` in eval mode.
        instance: The instance to attack.
        budget: Maximum number of cells to flip.
        objective: ``"optimality"`` (maximise the Opt ratio) or
            ``"efficiency"`` (maximise the Exp ratio subject to the Opt ratio
            staying at or below `opt_cap`).
        candidate_pool: Cells the search may choose from.
        baselines: Shared `Baseline` cache, as for `evaluate_alterations`.
        batch_size: Planner batch size per round.
        opt_cap: Opt ratio cap for the efficiency objective.

    Returns:
        list[dict]: One results row per round, i.e. the chosen set at budgets
        1, 2, … up to `budget`. Each row is a normal alteration row with
        ``method="greedy"`` and the objective recorded, so the rows append
        directly to the results manifest. Rounds that find no admissible
        addition stop the search early and shorten the list.

    Raises:
        ValueError: If `objective` is unknown.
    """
    if objective not in ("optimality", "efficiency"):
        raise ValueError(f"unknown objective {objective!r}")
    baselines = {} if baselines is None else baselines

    chosen: List[RC] = []
    history: List[Dict[str, object]] = []
    pool = [tuple(rc) for rc in candidate_pool]

    for _ in range(budget):
        requests = []
        for candidate in pool:
            if candidate in chosen:
                continue
            cells = tuple(sorted(chosen + [candidate]))
            altered = flip_cells(instance.map_design, cells)
            if not is_reachable(altered, instance.start_rc, instance.goal_rc):
                continue
            requests.append(
                AlterationRequest(
                    instance=instance,
                    flip_cells=cells,
                    method="greedy",
                    objective=objective,
                    extra={"candidate": list(candidate)},
                )
            )
        if not requests:
            break

        rows = evaluate_alterations(
            planner, requests, baselines=baselines, batch_size=batch_size
        )
        if objective == "optimality":
            best = max(rows, key=lambda r: float(r["opt_C1O1"]))
        else:
            eligible = [r for r in rows if float(r["opt_C1O1"]) <= opt_cap]
            if not eligible:
                break
            best = max(eligible, key=lambda r: float(r["exp_C1O1"]))

        chosen = [tuple(rc) for rc in best["flip_cells"]]
        best.pop("candidate", None)
        history.append(best)
    return history


def candidate_pool_from_rows(
    rows: Sequence[Dict[str, object]],
    objective: str,
    size: int = 64,
) -> List[RC]:
    """Rank k=1 sweep rows and return the most promising cells.

    Args:
        rows: Exhaustive k=1 rows for a single instance.
        objective: ``"optimality"`` or ``"efficiency"``; decides the ranking
            metric.
        size: How many cells to keep.

    Returns:
        list[tuple[int, int]]: The `size` best single cells, best first. Ties
        break by row order, which is row-major by cell, so the pool is
        deterministic.

    Raises:
        ValueError: If `objective` is unknown.
    """
    if objective == "optimality":
        key = "opt_C1O1"
    elif objective == "efficiency":
        key = "exp_C1O1"
    else:
        raise ValueError(f"unknown objective {objective!r}")
    ranked = sorted(rows, key=lambda r: -float(r[key]))
    return [tuple(r["flip_cells"][0]) for r in ranked[:size]]
