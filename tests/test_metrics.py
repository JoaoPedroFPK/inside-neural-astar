"""Unit tests for `src.metrics` (the project requires these).

All inputs are synthetic, so the expected values are computed by hand and the
tests pin the *definitions*, not the planner's behaviour.
"""

from __future__ import annotations

import numpy as np

from src.metrics import (
    exp_ratio,
    harmonic_mean,
    mediation_fraction,
    n_expansions,
    n_path_cells,
    opt_ratio,
    p_exp_reduction,
    p_opt_binary,
    path_cost,
)


def _map_with(n_ones: int, batch: int = 1) -> np.ndarray:
    """Return a ``(batch, 1, 4, 4)`` binary map with `n_ones` cells set."""
    out = np.zeros((batch, 1, 4, 4), dtype=np.float32)
    flat = out.reshape(batch, -1)
    flat[:, :n_ones] = 1.0
    return out


def test_path_cell_and_cost_counting() -> None:
    """A path of n cells is a path of n-1 unit-cost moves."""
    assert n_path_cells(_map_with(5)) == np.array([5.0])
    assert path_cost(_map_with(5)) == np.array([4.0])
    # An empty path map (search failure) reports -1, never 0 moves.
    assert path_cost(_map_with(0)) == np.array([-1.0])


def test_expansion_counting() -> None:
    """Expansions are the number of cells marked in the history map."""
    assert n_expansions(_map_with(7)) == np.array([7.0])


def test_p_opt_binary_is_equality_of_path_lengths() -> None:
    """p_opt is 1 only when the planner matches classical A* cell for cell."""
    vanilla = np.concatenate([_map_with(5), _map_with(5)])
    planner = np.concatenate([_map_with(5), _map_with(6)])
    assert list(p_opt_binary(vanilla, planner)) == [1.0, 0.0]


def test_p_exp_reduction_is_floored_at_zero() -> None:
    """p_exp is the relative reduction in expansions, never negative."""
    vanilla = np.concatenate([_map_with(10), _map_with(10)])
    planner = np.concatenate([_map_with(4), _map_with(16)])
    reductions = p_exp_reduction(vanilla, planner)
    assert np.allclose(reductions, [0.6, 0.0])


def test_harmonic_mean_matches_upstream_formula() -> None:
    """Hmean of 1.0 and 0.5 is 2/3; a zero component collapses it."""
    assert abs(harmonic_mean(1.0, 0.5) - 2.0 / 3.0) < 1e-9
    assert harmonic_mean(1.0, 0.0) < 1e-9


def test_opt_ratio_definition() -> None:
    """Opt is planner cost over the optimum of the same map; 1.0 = optimal."""
    ratios = opt_ratio(np.array([10.0, 12.0, -1.0]), np.array([10.0, 10.0, 10.0]))
    assert np.allclose(ratios[:2], [1.0, 1.2])
    assert np.isinf(ratios[2]), "an empty path must not look like a cheap path"
    # Unsolvable instances are nan, not a number that would pollute averages.
    assert np.isnan(opt_ratio(np.array([5.0]), np.array([np.inf]))[0])


def test_exp_ratio_is_the_complement_of_p_exp_while_better_than_classical() -> None:
    """Exp is a ratio to classical A*; above 1.0 means worse, which p_exp hides."""
    planner = np.array([4.0, 16.0])
    classical = np.array([10.0, 10.0])
    ratios = exp_ratio(planner, classical)
    assert np.allclose(ratios, [0.4, 1.6])
    reductions = p_exp_reduction(_map_with(10, batch=2), _map_with(4, batch=2))
    assert np.allclose(1.0 - reductions[0], ratios[0])


def test_mediation_fraction_definition() -> None:
    """Mediation fraction follows the project's definition, and is nan when there is no damage."""
    # Baseline 1.0, full alteration 1.4, encoder-only 1.3 -> 75% of the damage.
    assert abs(mediation_fraction(1.0, 1.3, 1.4) - 0.75) < 1e-9
    # Routes need not sum to 1: an interaction can overshoot.
    assert mediation_fraction(1.0, 1.5, 1.4) > 1.0
    assert np.isnan(mediation_fraction(1.0, 1.0, 1.0))


class _FakeOutput:
    """Minimal stand-in for `AstarOutput` with hand-made path/history maps."""

    def __init__(self, path_cells, history_cells):
        self.paths = np.stack([_map_with(n)[0] for n in path_cells])
        self.histories = np.stack([_map_with(n)[0] for n in history_cells])


def test_opt_binary_respects_tolerance() -> None:
    """Binary optimality is exact by default and can be given slack."""
    from src.metrics import opt_binary

    ratios = np.array([1.0, 1.04, 1.2, np.nan])
    assert list(opt_binary(ratios)) == [1.0, 0.0, 0.0, 0.0]
    assert list(opt_binary(ratios, tol=0.05)) == [1.0, 1.0, 0.0, 0.0]


def test_project_metrics_aggregates_both_families() -> None:
    """`project_metrics` reports the ratios and the paper's floored summaries.

    Two instances: one optimal and twice as efficient as classical A*, one 20 %
    too long and 60 % *less* efficient. The floored ``p_exp`` ignores the second
    instance's regression, ``mean_exp_ratio`` does not - the whole reason both
    are reported.
    """
    from src.metrics import project_metrics

    outputs = _FakeOutput(path_cells=[11, 13], history_cells=[5, 16])
    metrics = project_metrics(
        outputs,
        optimal_costs=np.array([10.0, 10.0]),
        classical_expansions=np.array([10.0, 10.0]),
    )

    assert np.allclose(metrics["opt_ratio"], [1.0, 1.2])
    assert list(metrics["opt_binary"]) == [1.0, 0.0]
    assert np.allclose(metrics["exp_ratio"], [0.5, 1.6])

    summary = metrics["summary"]
    assert summary["n"] == 2
    assert abs(summary["p_opt"] - 0.5) < 1e-12
    assert abs(summary["mean_exp_ratio"] - 1.05) < 1e-12
    assert abs(summary["p_exp"] - 0.25) < 1e-12  # mean(max(1-r, 0)) = (0.5 + 0) / 2
    assert summary["n_worse_than_classical"] == 1
    assert abs(summary["h_mean"] - harmonic_mean(0.5, 0.25)) < 1e-12


def test_p_exp_is_recoverable_from_exp_ratio() -> None:
    """The paper's p_exp equals mean(max(1 - exp_ratio, 0)) by construction.

    Pins the bridge between the two metric families, so the E0 line and the
    perturbation tables can be compared without re-deriving it.
    """
    from src.metrics import exp_ratio as ratio_fn

    vanilla = np.stack([_map_with(n)[0] for n in (10, 10, 10)])
    planner = np.stack([_map_with(n)[0] for n in (4, 10, 16)])
    reductions = p_exp_reduction(vanilla, planner)
    ratios = ratio_fn(np.array([4.0, 10.0, 16.0]), np.array([10.0, 10.0, 10.0]))
    assert np.allclose(reductions, np.maximum(1.0 - ratios, 0.0))
