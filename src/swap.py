"""The 2x2 cost/obstacle swap harness — E3 (PLAN.md step 7).

The question this answers: when a small map change makes the planner fail,
*where* did the failure come from — the encoder producing distorted costs, or
the search amplifying almost-correct ones?

The design exploits the fact verified in step 1: the search takes the cost map
and the obstacle mask as independent inputs, so they may come from different
maps. Four runs per alteration:

======  ==================  ====================  =========================
label   cost map from       obstacle mask from    isolates
======  ==================  ====================  =========================
C0O0    original map        original map          baseline, no change
C0O1    original map        altered map           the search route
C1O0    altered map         original map          the encoder route
C1O1    altered map         altered map           the full alteration
======  ==================  ====================  =========================

In mediation terms the flip is the treatment, the cost map is the mediator,
and path quality (or search effort) is the outcome: ``C1O0`` is the effect
*through* the mediator, ``C0O1`` the effect that bypasses it.

**The reference depends on the obstacle mask, not the cost map.** The mask
defines the problem actually being solved, so a condition running on the
original walls is scored against the original map's optimum and classical-A\\*
effort, and a condition running on altered walls against the altered map's.
Getting this backwards silently biases every mediation fraction.

A known weakness, recorded rather than hidden: the off-diagonal pairs are
off-distribution by construction — the encoder never produced *that* cost map
for *that* mask. `src.perturb` provides an independent cross-check that does
not share this weakness, since the encoder's response to a single flip is
confined to an 11x11 window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .data import MazeInstance
from .exact_solver import solve_astar, solve_dijkstra
from .grid import RC, flip_cells, validate_path_map
from .metrics import exp_ratio, mediation_fraction, n_expansions, opt_ratio, path_cost
from .planner import encode_cost_map, run_astar_module

#: The four conditions, in the project's canonical order, as
#: ``(label, costs_from_altered, mask_from_altered)``.
CONDITIONS: Tuple[Tuple[str, bool, bool], ...] = (
    ("C0O0", False, False),
    ("C0O1", False, True),
    ("C1O0", True, False),
    ("C1O1", True, True),
)


@dataclass
class SwapResult:
    """Metrics for one alteration under all four conditions.

    Attributes:
        instance_id: Instance the alteration belongs to.
        flip_cells: The flipped cells.
        opt: Opt ratio per condition label.
        exp: Exp ratio per condition label.
        path_cost: Raw path cost (moves) per condition label.
        expansions: Raw expansion count per condition label.
        valid: Whether each condition returned a legal path for its own mask.
        dcost_l1: Total absolute change in the cost map caused by the flip.
        dcost_frac_under_flips: Share of that change sitting on the flipped
            cells themselves.
        dcost_frac_outside_receptive_field: Share of the change further than
            the encoder's receptive-field radius from any flipped cell. Must be
            0; a non-zero value means the locality property has broken.
    """

    instance_id: int
    flip_cells: Tuple[RC, ...]
    opt: Dict[str, float]
    exp: Dict[str, float]
    path_cost: Dict[str, float]
    expansions: Dict[str, float]
    valid: Dict[str, bool]
    dcost_l1: float
    dcost_frac_under_flips: float
    dcost_frac_outside_receptive_field: float

    def mediation_raw(self, metric: str = "opt") -> Dict[str, float]:
        """Mediation fractions on the **raw** counts rather than the ratios.

        `mediation` decomposes the normalised metrics, whose denominators are
        the altered map's own optimum and classical-A\* effort. Those
        denominators differ between the original-mask and altered-mask
        conditions — by about one expanded cell at k=1 — so a small part of any
        measured effect is the yardstick moving rather than the planner. This
        variant uses the planner's raw path length and expansion count, which
        has no such confound but is not comparable across instances.

        Agreement between the two is the robustness check; disagreement means
        the normalisation is doing the work.

        Args:
            metric: ``"opt"`` (path length in moves) or ``"exp"`` (expanded
                cells).

        Returns:
            dict[str, float]: ``encoder``, ``search`` and ``total_effect``.

        Raises:
            ValueError: If `metric` is not recognised.
        """
        if metric not in ("opt", "exp"):
            raise ValueError(f"unknown metric {metric!r}; expected 'opt' or 'exp'")
        values = self.path_cost if metric == "opt" else self.expansions
        return {
            "encoder": mediation_fraction(values["C0O0"], values["C1O0"], values["C1O1"]),
            "search": mediation_fraction(values["C0O0"], values["C0O1"], values["C1O1"]),
            "total_effect": values["C1O1"] - values["C0O0"],
        }

    def mediation(self, metric: str = "opt") -> Dict[str, float]:
        """Mediation fractions for one metric.

        Args:
            metric: ``"opt"`` or ``"exp"``.

        Returns:
            dict[str, float]: ``encoder`` and ``search`` fractions, plus
            ``total_effect`` (the denominator). Both fractions are ``nan`` when
            the full alteration caused no change in the metric — there is no
            damage to decompose, which is a reportable outcome rather than an
            error.

        Raises:
            ValueError: If `metric` is not recognised.
        """
        if metric not in ("opt", "exp"):
            raise ValueError(f"unknown metric {metric!r}; expected 'opt' or 'exp'")
        values = self.opt if metric == "opt" else self.exp
        return {
            "encoder": mediation_fraction(values["C0O0"], values["C1O0"], values["C1O1"]),
            "search": mediation_fraction(values["C0O0"], values["C0O1"], values["C1O1"]),
            "total_effect": values["C1O1"] - values["C0O0"],
        }


def run_swap(
    planner,
    instance: MazeInstance,
    cells: Sequence[RC],
    receptive_radius: int = 5,
) -> SwapResult:
    """Evaluate one alteration under all four cost/obstacle conditions.

    Args:
        planner: A `NeuralAstar` in eval mode.
        instance: The instance to alter.
        cells: Cells to flip.
        receptive_radius: The encoder's receptive-field radius, used only for
            the locality diagnostic (5 for the pretrained CNN encoder).

    Returns:
        SwapResult: Metrics for every condition plus the ΔCost diagnostics.

    Raises:
        ValueError: If the flips disconnect the goal from the start.
    """
    from .grid import is_reachable

    original = instance.map_design
    altered = flip_cells(original, cells)
    if not is_reachable(altered, instance.start_rc, instance.goal_rc):
        raise ValueError(
            f"instance {instance.instance_id}: flips {tuple(cells)} disconnect the goal"
        )

    maps0, starts, goals = instance.to_tensors()
    maps1 = torch.from_numpy(altered).reshape_as(maps0)
    with torch.no_grad():
        costs0 = encode_cost_map(planner, maps0, starts, goals)
        costs1 = encode_cost_map(planner, maps1, starts, goals)

    # References come from the obstacle mask, which defines the problem.
    references = {}
    for tag, mask_array in (("O0", original), ("O1", altered)):
        references[tag] = (
            float(solve_dijkstra(mask_array, instance.start_rc, instance.goal_rc).cost),
            float(solve_astar(mask_array, instance.start_rc, instance.goal_rc).expansions),
        )

    opt: Dict[str, float] = {}
    exp: Dict[str, float] = {}
    costs_out: Dict[str, float] = {}
    expansions_out: Dict[str, float] = {}
    valid: Dict[str, bool] = {}
    for label, costs_altered, mask_altered in CONDITIONS:
        cost_tensor = costs1 if costs_altered else costs0
        mask_tensor = maps1 if mask_altered else maps0
        mask_array = altered if mask_altered else original
        with torch.no_grad():
            out = run_astar_module(planner, cost_tensor, starts, goals, mask_tensor)
        optimum, classical = references["O1" if mask_altered else "O0"]
        moves = float(path_cost(out.paths)[0])
        expanded = float(n_expansions(out.histories)[0])
        opt[label] = float(opt_ratio(np.array([moves]), np.array([optimum]))[0])
        exp[label] = float(exp_ratio(np.array([expanded]), np.array([classical]))[0])
        costs_out[label] = moves
        expansions_out[label] = expanded
        valid[label] = validate_path_map(
            out.paths[0, 0].numpy(), mask_array, instance.start_rc, instance.goal_rc
        )[0]

    delta = np.abs(costs1[0, 0].numpy() - costs0[0, 0].numpy())
    total = float(delta.sum())
    under = float(sum(delta[rc] for rc in cells))
    rows, columns = np.ogrid[: delta.shape[0], : delta.shape[1]]
    far = np.ones_like(delta, dtype=bool)
    for rc in cells:
        far &= (np.abs(rows - rc[0]) > receptive_radius) | (
            np.abs(columns - rc[1]) > receptive_radius
        )
    return SwapResult(
        instance_id=instance.instance_id,
        flip_cells=tuple(cells),
        opt=opt,
        exp=exp,
        path_cost=costs_out,
        expansions=expansions_out,
        valid=valid,
        dcost_l1=total,
        dcost_frac_under_flips=(under / total) if total > 0 else float("nan"),
        dcost_frac_outside_receptive_field=(float(delta[far].sum()) / total)
        if total > 0
        else float("nan"),
    )


def swap_columns(result: SwapResult) -> Dict[str, object]:
    """Flatten a `SwapResult` into the results-manifest columns.

    Args:
        result: One swap evaluation.

    Returns:
        dict: ``opt_C0O0`` … ``exp_C1O1``, the mediation fractions for both
        metrics, and the ΔCost diagnostics, ready to merge into the manifest.
    """
    out: Dict[str, object] = {}
    for label in ("C0O0", "C0O1", "C1O0", "C1O1"):
        out[f"opt_{label}"] = result.opt[label]
        out[f"exp_{label}"] = result.exp[label]
        out[f"path_cost_{label}"] = result.path_cost[label]
        out[f"expansions_{label}"] = result.expansions[label]
        out[f"valid_{label}"] = result.valid[label]
    for metric in ("opt", "exp"):
        fractions = result.mediation(metric)
        out[f"mediation_{metric}_encoder"] = fractions["encoder"]
        out[f"mediation_{metric}_search"] = fractions["search"]
        out[f"total_effect_{metric}"] = fractions["total_effect"]
        raw = result.mediation_raw(metric)
        out[f"mediation_{metric}_encoder_raw"] = raw["encoder"]
        out[f"mediation_{metric}_search_raw"] = raw["search"]
        out[f"total_effect_{metric}_raw"] = raw["total_effect"]
    out["dcost_l1"] = result.dcost_l1
    out["dcost_frac_under_flips"] = result.dcost_frac_under_flips
    out["dcost_frac_outside_receptive_field"] = result.dcost_frac_outside_receptive_field
    return out


def classify_route(
    fractions: Dict[str, float],
    tolerance: float = 0.25,
) -> str:
    """Label one alteration's mediation pattern.

    Args:
        fractions: Output of `SwapResult.mediation`.
        tolerance: How far from 0 a route's fraction may sit while still
            counting as "did not reproduce the damage", and how far below 1 it
            may sit while counting as "reproduced it".

    Returns:
        str: One of

        * ``"no damage"`` — the full alteration did not make the metric worse
          (no change, or an improvement). There is nothing to decompose, and
          the fractions would be meaningless: dividing by a negative total
          flips the sign of both routes;
        * ``"encoder"`` — only the encoder route reproduces the damage;
        * ``"search"`` — only the search route does;
        * ``"both"`` — each route reproduces most of it on its own;
        * ``"interaction"`` — neither does, so the two are needed together.

    Note:
        The labels are a reading aid for the distribution, not a statistical
        test. The underlying fractions are what the report plots; a tolerance
        of 0.25 is arbitrary and is stated wherever the labels appear.
    """
    if not np.isfinite(fractions["total_effect"]) or fractions["total_effect"] <= 0:
        return "no damage"
    encoder = fractions["encoder"]
    search = fractions["search"]
    encoder_reproduces = np.isfinite(encoder) and encoder >= 1.0 - tolerance
    search_reproduces = np.isfinite(search) and search >= 1.0 - tolerance
    if encoder_reproduces and search_reproduces:
        return "both"
    if encoder_reproduces:
        return "encoder"
    if search_reproduces:
        return "search"
    return "interaction"


def summarize_swaps(
    results: Sequence[SwapResult],
    metric: str = "opt",
    tolerance: float = 0.25,
    raw: bool = False,
) -> Dict[str, object]:
    """Aggregate swap results into the step-7 headline numbers.

    Args:
        results: Swap evaluations.
        metric: ``"opt"`` or ``"exp"``.
        tolerance: Passed to `classify_route`.
        raw: Decompose the raw counts (`SwapResult.mediation_raw`) instead of
            the normalised ratios. Reported alongside as a robustness check.

    Returns:
        dict: Counts per route label, the median and mean of each route's
        fraction over alterations that actually caused damage, and the number
        of alterations with no damage to decompose.
    """
    labels: Dict[str, int] = {}
    encoder_values: List[float] = []
    search_values: List[float] = []
    for result in results:
        fractions = result.mediation_raw(metric) if raw else result.mediation(metric)
        label = classify_route(fractions, tolerance)
        labels[label] = labels.get(label, 0) + 1
        if label != "no damage":
            encoder_values.append(fractions["encoder"])
            search_values.append(fractions["search"])

    encoder = np.array(encoder_values, dtype=np.float64)
    search = np.array(search_values, dtype=np.float64)
    return {
        "metric": metric,
        "raw": raw,
        "n": len(results),
        "labels": labels,
        "n_with_damage": int(len(encoder)),
        "encoder_median": float(np.nanmedian(encoder)) if encoder.size else float("nan"),
        "encoder_mean": float(np.nanmean(encoder)) if encoder.size else float("nan"),
        "search_median": float(np.nanmedian(search)) if search.size else float("nan"),
        "search_mean": float(np.nanmean(search)) if search.size else float("nan"),
    }
