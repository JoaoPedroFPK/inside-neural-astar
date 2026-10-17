"""Metrics on planner outputs.

Two families live here, and they must not be confused:

**(a) The paper's metrics** (`p_opt`, `p_exp`, `h_mean`). These replicate
`neural_astar.utils.training.PlannerModule.validation_step` line for line, so
that E0 can be compared against the numbers reported in the ICML'21 paper.
``p_opt`` is *binary* (did the planner's path have the same length as classical
A*'s?) and ``p_exp`` is a *reduction* (how many fewer nodes did it expand,
relative to classical A*, floored at 0).

**(b) This project's metrics** (`opt_ratio`, `exp_ratio`), as defined in the
project's definitions: a path-cost ratio to the true optimum of the *same* map (1.0 =
optimal) and an expansion count ratio to classical A* on the same map. Under
perturbation these are always measured against the **altered** map's optimum,
which is why they need the exact solver of PLAN.md step 2; the functions here
take that optimum as an argument rather than computing it.

Path costs are counted in moves under the unit-cost model (every one of the
eight moves costs 1, see `src.grid`). A path map with ``n`` marked cells is a
path of ``n - 1`` moves, so comparing cell counts and comparing costs are the
same comparison — which is why (a) can be computed from path maps alone.
"""

from __future__ import annotations

from typing import Dict, Union

import numpy as np

Array = Union[np.ndarray, "torch.Tensor"]  # noqa: F821


def _to_numpy(x: Array) -> np.ndarray:
    """Return `x` as a detached numpy array.

    Args:
        x: A numpy array or a torch tensor (possibly requiring grad).

    Returns:
        np.ndarray: The same data as numpy, never sharing autograd state.
    """
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def n_path_cells(paths: Array) -> np.ndarray:
    """Number of cells marked on each path map of a batch.

    Args:
        paths: ``(B, 1, H, W)`` (or ``(B, H, W)``) binary path maps, as
            returned in `AstarOutput.paths`.

    Returns:
        np.ndarray: Float array of shape ``(B,)``.
    """
    arr = _to_numpy(paths)
    return arr.reshape(arr.shape[0], -1).sum(axis=1).astype(np.float64)


def path_cost(paths: Array) -> np.ndarray:
    """Unit-cost length (in moves) of each path in a batch.

    Args:
        paths: ``(B, 1, H, W)`` binary path maps.

    Returns:
        np.ndarray: ``n_path_cells(paths) - 1``, i.e. the number of moves.
        A degenerate path map with zero cells (search failure) yields ``-1``,
        which callers should treat as "no path".
    """
    return n_path_cells(paths) - 1.0


def n_expansions(histories: Array) -> np.ndarray:
    """Number of nodes expanded by the search, per instance.

    Args:
        histories: ``(B, 1, H, W)`` binary maps from `AstarOutput.histories`,
            which hold 1 for every node that was selected (closed) at some
            step.

    Returns:
        np.ndarray: Float array of shape ``(B,)``.
    """
    arr = _to_numpy(histories)
    return arr.reshape(arr.shape[0], -1).sum(axis=1).astype(np.float64)


def p_opt_binary(vanilla_paths: Array, planner_paths: Array) -> np.ndarray:
    """Per-instance binary optimality, exactly as the paper computes it.

    Args:
        vanilla_paths: ``(B, 1, H, W)`` path maps from classical A* on the same
            maps and the same start/goal pairs.
        planner_paths: ``(B, 1, H, W)`` path maps from the planner under test.

    Returns:
        np.ndarray: Float array of shape ``(B,)``, 1.0 where the two paths have
        the same number of cells (equivalently, the same unit cost) and 0.0
        otherwise.

    Note:
        This is a comparison against classical A*'s path *length*, not against
        a certified optimum. Classical A* with a consistent heuristic is
        optimal, so the two coincide on unaltered maps; after a perturbation,
        use `opt_ratio` against the exact solver instead (project rule:
        "An alteration counts as an error only if Neural A* deviates from the
        *altered map's* optimum").
    """
    return (n_path_cells(vanilla_paths) == n_path_cells(planner_paths)).astype(np.float64)


def p_exp_reduction(vanilla_histories: Array, planner_histories: Array) -> np.ndarray:
    """Per-instance search-efficiency gain, exactly as the paper computes it.

    Args:
        vanilla_histories: ``(B, 1, H, W)`` histories from classical A*.
        planner_histories: ``(B, 1, H, W)`` histories from the planner.

    Returns:
        np.ndarray: ``max((exp_astar - exp_planner) / exp_astar, 0)`` per
        instance, so 0.0 means "no better than classical A*" and 0.4 means
        "40% fewer nodes expanded".
    """
    exp_astar = n_expansions(vanilla_histories)
    exp_planner = n_expansions(planner_histories)
    return np.maximum((exp_astar - exp_planner) / exp_astar, 0.0)


def harmonic_mean(p_opt: float, p_exp: float, eps: float = 1e-10) -> float:
    """Harmonic mean of the two paper metrics (the paper's ``Hmean``).

    Args:
        p_opt: Mean binary optimality in ``[0, 1]``.
        p_exp: Mean expansion reduction in ``[0, 1]``.
        eps: Guard against division by zero, same value as upstream.

    Returns:
        float: ``2 / (1 / p_opt + 1 / p_exp)``.
    """
    return float(2.0 / (1.0 / (p_opt + eps) + 1.0 / (p_exp + eps)))


def paper_metrics(
    vanilla_outputs,
    planner_outputs,
) -> Dict[str, float]:
    """Aggregate the three paper metrics over a batch of outputs.

    Args:
        vanilla_outputs: `AstarOutput` from classical A*.
        planner_outputs: `AstarOutput` from the planner under test.

    Returns:
        dict[str, float]: ``{"p_opt": ..., "p_exp": ..., "h_mean": ...}``,
        each averaged over the batch (``h_mean`` computed from the averages,
        as upstream does, not averaged per instance).
    """
    p_opt = float(p_opt_binary(vanilla_outputs.paths, planner_outputs.paths).mean())
    p_exp = float(p_exp_reduction(vanilla_outputs.histories, planner_outputs.histories).mean())
    return {"p_opt": p_opt, "p_exp": p_exp, "h_mean": harmonic_mean(p_opt, p_exp)}


def opt_ratio(planner_path_cost: Array, optimal_cost: Array) -> np.ndarray:
    """Path-cost ratio to the true optimum of the same map (1.0 = optimal).

    This is the project's ``Opt``. Values below 1.0
    are impossible and indicate that `optimal_cost` was not the optimum of the
    map the planner actually searched.

    Args:
        planner_path_cost: Per-instance unit cost of the planner's path, e.g.
            from `path_cost`.
        optimal_cost: Per-instance optimal cost on the *same* map, from
            `src.exact_solver` (or, on unaltered maps, from the dataset's
            stored distances).

    Returns:
        np.ndarray: ``planner_path_cost / optimal_cost``, with ``np.inf``
        where the planner returned no path and ``np.nan`` where the instance
        is unsolvable (optimal cost not finite).
    """
    planner = np.asarray(_to_numpy(planner_path_cost), dtype=np.float64)
    optimum = np.asarray(_to_numpy(optimal_cost), dtype=np.float64)
    out = np.full(planner.shape, np.nan, dtype=np.float64)
    solvable = np.isfinite(optimum) & (optimum > 0)
    out[solvable] = planner[solvable] / optimum[solvable]
    out[solvable & (planner < 0)] = np.inf  # empty path map = search failure
    return out


def exp_ratio(planner_expansions: Array, classical_expansions: Array) -> np.ndarray:
    """Expansion-count ratio to classical A* on the same map.

    This is the project's ``Exp``, which is a
    *ratio*, deliberately different from the paper's ``p_exp`` reduction:
    ``exp_ratio = 1 - p_exp`` only while the planner is at least as efficient
    as classical A*; above 1.0 the planner is worse, which `p_exp` floors away.

    Args:
        planner_expansions: Per-instance expansion counts of the planner.
        classical_expansions: Per-instance expansion counts of classical A* on
            the same map, same start and goal, same heuristic.

    Returns:
        np.ndarray: ``planner_expansions / classical_expansions``, ``np.nan``
        where the classical count is not positive.
    """
    planner = np.asarray(_to_numpy(planner_expansions), dtype=np.float64)
    classical = np.asarray(_to_numpy(classical_expansions), dtype=np.float64)
    out = np.full(planner.shape, np.nan, dtype=np.float64)
    ok = classical > 0
    out[ok] = planner[ok] / classical[ok]
    return out


def mediation_fraction(m_baseline: float, m_single: float, m_full: float) -> float:
    """Share of a metric's damage reproduced by one swap condition.

    Implements the project's definition: for a metric ``M``, the encoder route is
    ``(M[C1O0] - M[C0O0]) / (M[C1O1] - M[C0O0])`` and the search route is the
    same with ``C0O1``. Both are reported; they need not sum to 1.

    Args:
        m_baseline: ``M`` under ``C0O0`` (original costs, original obstacles).
        m_single: ``M`` under the single-route condition (``C1O0`` for the
            encoder route, ``C0O1`` for the search route).
        m_full: ``M`` under ``C1O1`` (the full alteration).

    Returns:
        float: The mediation fraction, or ``np.nan`` if the full alteration
        caused no change in ``M`` (denominator zero — there is no damage to
        decompose, which is itself a reportable outcome).
    """
    denominator = m_full - m_baseline
    if denominator == 0:
        return float("nan")
    return float((m_single - m_baseline) / denominator)


def opt_binary(ratios: Array, tol: float = 0.0) -> np.ndarray:
    """Binary optimality derived from `opt_ratio` values.

    Equivalent to the paper's `p_opt_binary` but anchored on a certified
    optimum rather than on classical A\\*'s path, so it stays meaningful on
    altered maps.

    Args:
        ratios: Per-instance `opt_ratio` values.
        tol: Slack on the ratio. The default 0.0 demands exact optimality,
            which is the right default under unit costs (all path costs are
            integers). The efficiency objective uses a 1.05 cap
            elsewhere; pass ``tol=0.05`` to mirror it.

    Returns:
        np.ndarray: 1.0 where ``ratio <= 1 + tol``, else 0.0. ``nan`` ratios
        (unsolvable instances) yield 0.0 and should be masked out beforehand if
        they are expected.
    """
    arr = np.asarray(_to_numpy(ratios), dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return (arr <= 1.0 + tol).astype(np.float64)


def project_metrics(
    planner_outputs,
    optimal_costs: Array,
    classical_expansions: Array,
    tol: float = 0.0,
) -> Dict[str, object]:
    """Aggregate this project's metrics for a batch of planner outputs.

    This is the step-2 entry point: it takes a planner output plus the two
    quantities that must come from the map being evaluated — the certified
    optimum (`src.exact_solver.solve_dijkstra`) and classical A\\*'s expansion
    count on that same map (`src.exact_solver.solve_astar`, or upstream's
    `VanillaAstar`, which it reproduces exactly).

    Args:
        planner_outputs: `AstarOutput` from the planner under test.
        optimal_costs: Per-instance optimal path cost **on the same map** the
            planner searched. After a perturbation this is the *altered* map's
            optimum.
        classical_expansions: Per-instance expansion count of classical A\\* on
            that same map.
        tol: Slack passed to `opt_binary`.

    Returns:
        dict: Per-instance arrays under ``"opt_ratio"``, ``"opt_binary"``,
        ``"exp_ratio"``, ``"path_cost"``, ``"expansions"``, and scalar means
        under ``"summary"``: ``mean_opt_ratio``, ``p_opt``, ``mean_exp_ratio``,
        ``p_exp`` and ``h_mean``.

    Note:
        ``p_exp`` and ``h_mean`` in the summary are the *paper's* quantities
        (``p_exp = mean(max(1 - exp_ratio, 0))``), recomputed here from the
        ratios so that one call can be compared against both the E0 line and
        the perturbation tables. They floor instances where the planner is
        worse than classical A\\*; ``mean_exp_ratio`` does not, which is why
        the two disagree whenever any instance exceeds 1.0.
    """
    costs = path_cost(planner_outputs.paths)
    expansions = n_expansions(planner_outputs.histories)
    ratios_opt = opt_ratio(costs, optimal_costs)
    ratios_exp = exp_ratio(expansions, classical_expansions)
    flags = opt_binary(ratios_opt, tol=tol)

    p_opt = float(np.nanmean(flags))
    p_exp = float(np.nanmean(np.maximum(1.0 - ratios_exp, 0.0)))
    return {
        "path_cost": costs,
        "expansions": expansions,
        "opt_ratio": ratios_opt,
        "opt_binary": flags,
        "exp_ratio": ratios_exp,
        "summary": {
            "n": int(costs.size),
            "mean_opt_ratio": float(np.nanmean(ratios_opt)),
            "p_opt": p_opt,
            "mean_exp_ratio": float(np.nanmean(ratios_exp)),
            "p_exp": p_exp,
            "h_mean": harmonic_mean(p_opt, p_exp),
            "n_worse_than_classical": int(np.nansum(ratios_exp > 1.0)),
        },
    }
