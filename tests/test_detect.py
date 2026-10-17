"""PLAN.md step 12: the altered-map detector and its two baselines."""

from __future__ import annotations

import numpy as np
import pytest

from src.detect import (
    ACTIVATION_LAYERS,
    MAP_ONLY_FEATURES,
    RESIDUAL_FEATURES,
    MahalanobisScorer,
    bootstrap_auc_gap,
    build_examples,
    evaluate_detector,
    evaluate_mahalanobis,
    fit_detector,
    map_only_features,
    paired_alteration,
    pooled_activations,
    residual_statistics,
)
from src.grid import flip_cells


@pytest.fixture(scope="module")
def instance(test_split):
    """One test instance with a fixed start."""
    return test_split.instance(0, seed=0)


def test_residual_statistics_returns_every_declared_feature() -> None:
    """The feature vector's order is part of the module's contract."""
    stats = residual_statistics(np.zeros((32, 32)), np.ones((32, 32), bool))
    assert tuple(stats) == RESIDUAL_FEATURES


def test_a_zero_residual_scores_zero_everywhere() -> None:
    """A surrogate that predicts the cost map exactly leaves nothing to detect."""
    stats = residual_statistics(np.zeros((32, 32)), np.ones((32, 32), bool), clean_sd=0.1)
    assert all(value == 0.0 for value in stats.values())


def test_an_empty_mask_does_not_divide_by_zero() -> None:
    """An instance whose goal is unreachable has no usable cells."""
    stats = residual_statistics(np.ones((32, 32)), np.zeros((32, 32), bool))
    assert all(value == 0.0 for value in stats.values())


def test_a_localised_residual_concentrates_more_than_a_diffuse_one() -> None:
    """`max_window_share` is the feature meant to separate these two shapes."""
    mask = np.ones((32, 32), bool)
    localised = np.zeros((32, 32))
    localised[14:17, 14:17] = 1.0
    diffuse = np.abs(np.random.default_rng(0).normal(0.0, 0.3, (32, 32)))

    assert residual_statistics(localised, mask)["max_window_share"] == pytest.approx(1.0)
    assert residual_statistics(diffuse, mask)["max_window_share"] < 0.3


def test_the_outlier_count_is_measured_against_the_clean_scale() -> None:
    """Passing the clean spread is what makes this feature informative.

    Without it the threshold is the map's own spread, so a residual that is
    uniformly far from zero — the encoder off by a constant everywhere — looks
    exactly like a quiet one.
    """
    mask = np.ones((32, 32), bool)
    residual = np.full((32, 32), 0.5)

    assert residual_statistics(residual, mask, clean_sd=0.05)["frac_over_2sd"] == 1.0
    assert residual_statistics(residual, mask)["frac_over_2sd"] == 0.0


def test_mahalanobis_scores_an_outlier_above_the_distribution_centre() -> None:
    """The baseline must at least do the thing it is named for."""
    rng = np.random.default_rng(0)
    scorer = MahalanobisScorer.fit(rng.normal(size=(60, 8)))
    assert scorer.score(np.full(8, 6.0)) > scorer.score(np.zeros(8))


def test_mahalanobis_needs_more_than_one_clean_example() -> None:
    """A covariance cannot be estimated from a single point."""
    with pytest.raises(ValueError, match="at least two"):
        MahalanobisScorer.fit([np.zeros(4)])


def test_map_only_features_returns_every_declared_feature(instance) -> None:
    """The control's feature order is part of its contract."""
    plain = map_only_features(instance, instance.map_design)
    assert tuple(plain) == MAP_ONLY_FEATURES


def test_blocking_an_open_cell_leaves_exactly_one_isolated_obstacle(instance) -> None:
    """This is the control's main signal, so it must count what it claims.

    A free cell all of whose eight neighbours are free becomes, when flipped,
    an obstacle all of whose eight neighbours are free — conspicuous without
    any model.
    """
    from scipy.ndimage import convolve

    free = instance.map_design > 0
    kernel = np.ones((3, 3))
    kernel[1, 1] = 0
    neighbours = convolve(free.astype(np.float64), kernel, mode="constant", cval=0.0)
    open_cells = np.argwhere(free & (neighbours == 8))
    if not len(open_cells):
        pytest.skip("this instance has no cell in fully open ground")
    cell = (int(open_cells[0][0]), int(open_cells[0][1]))

    before = map_only_features(instance, instance.map_design)
    after = map_only_features(instance, flip_cells(instance.map_design, [cell]))
    assert after["isolated_obstacles"] == before["isolated_obstacles"] + 1
    assert after["obstacle_fraction"] > before["obstacle_fraction"]


def test_the_control_reads_the_same_optimum_as_the_dataset(instance) -> None:
    """`optimum_length` must be the real path length, not a proxy."""
    plain = map_only_features(instance, instance.map_design)
    assert plain["optimum_length"] == pytest.approx(instance.optimal_cost)


def test_pooled_activations_covers_every_offered_layer(pretrained_planner, instance) -> None:
    """The baseline is reported at its best over these variants, so all must exist."""
    pooled = pooled_activations(pretrained_planner, instance, instance.map_design)
    expected = {f"l{layer}_{how}" for layer in ACTIVATION_LAYERS for how in ("mean", "max")}
    assert set(pooled) == expected
    # Channel counts follow the encoder stack: 32, 64, 128, 256.
    assert [len(pooled[f"l{layer}_mean"]) for layer in ACTIVATION_LAYERS] == [32, 64, 128, 256]


def test_pooled_activations_respond_to_the_altered_map(pretrained_planner, instance) -> None:
    """If the pooled vector ignored the flip, the baseline could not work at all."""
    clean = pooled_activations(pretrained_planner, instance, instance.map_design)
    altered = pooled_activations(
        pretrained_planner, instance, flip_cells(instance.map_design, [(16, 16)])
    )
    assert not np.allclose(clean["l2_mean"], altered["l2_mean"])


def test_pooled_activations_leave_no_hooks_behind(pretrained_planner, instance) -> None:
    """Forward hooks that outlive the call would corrupt every later forward pass."""
    pooled_activations(pretrained_planner, instance, instance.map_design)
    for layer in ACTIVATION_LAYERS:
        assert not pretrained_planner.encoder.model[layer]._forward_hooks


def test_build_examples_pairs_each_instance_and_keeps_rows_aligned(
    pretrained_planner, test_split
) -> None:
    """Every array it returns must be indexable by the same row number.

    The evaluation split is drawn by instance, so a misalignment between the
    residual rows, the activation rows and the labels would silently put one
    detector's clean map against another's altered one.
    """
    instances = [test_split.instance(i, seed=0) for i in range(3)]
    alterations = {inst.instance_id: [(16, 16)] for inst in instances}

    def builder(inst, design):
        """Two trivial features, enough to exercise the plumbing."""
        mask = design > 0
        return np.stack([design[mask], np.ones(mask.sum())], axis=1), mask

    class Constant:
        """Stand-in surrogate that predicts a constant cost everywhere."""

        class _Model:
            def predict(self, matrix):
                """Return the same value for every cell."""
                return np.full(len(matrix), 0.5)

        model = _Model()

    residual, activations, plain, labels, ids = build_examples(
        pretrained_planner, Constant(), instances, alterations, builder
    )
    assert residual.shape == (6, len(RESIDUAL_FEATURES))
    assert plain.shape == (6, len(MAP_ONLY_FEATURES))
    assert list(labels) == [0, 1, 0, 1, 0, 1]
    assert ids == [0, 0, 1, 1, 2, 2]
    for matrix in activations.values():
        assert len(matrix) == 6


def test_build_examples_skips_instances_with_no_alteration(
    pretrained_planner, test_split
) -> None:
    """A condition need not cover every instance; the rest must be dropped cleanly."""

    def builder(inst, design):
        """One trivial feature."""
        mask = design > 0
        return np.ones((int(mask.sum()), 1)), mask

    class Constant:
        class _Model:
            def predict(self, matrix):
                """Return zero for every cell."""
                return np.zeros(len(matrix))

        model = _Model()

    instances = [test_split.instance(i, seed=0) for i in range(3)]
    _, _, _, labels, ids = build_examples(
        pretrained_planner, Constant(), instances, {1: [(16, 16)]}, builder
    )
    assert ids == [1, 1]
    assert list(labels) == [0, 1]


def _separable_problem(signal: float, seed: int = 0):
    """Build a detection problem whose first feature carries `signal`."""
    rng = np.random.default_rng(seed)
    ids = np.repeat(np.arange(60), 2)
    labels = np.tile([0, 1], 60)
    informative = (labels * signal + rng.normal(0, 1.0, len(labels))).reshape(-1, 1)
    noise = rng.normal(0, 1.0, (len(labels), 1))
    is_train = ids < 30
    return informative, noise, labels, is_train, list(ids)


def test_the_detector_is_at_chance_on_pure_noise() -> None:
    """A sanity check on the scoring itself: no signal, no skill."""
    _, noise, labels, is_train, _ = _separable_problem(signal=0.0)
    scores = evaluate_detector(
        noise[is_train], labels[is_train], noise[~is_train], labels[~is_train]
    )
    assert 0.3 < scores["auc"] < 0.7


def test_the_bootstrap_finds_a_real_gap_and_excludes_zero() -> None:
    """An informative feature set must beat noise with an interval above zero."""
    informative, noise, labels, is_train, ids = _separable_problem(signal=3.0)
    gap = bootstrap_auc_gap(informative, noise, labels, is_train, ids, draws=400)
    assert gap["gap"] > 0.2
    assert gap["lo"] > 0.0
    assert gap["p_positive"] > 0.95


def test_the_bootstrap_reports_no_gap_between_identical_feature_sets() -> None:
    """The guard against reading a margin that is not there."""
    informative, _, labels, is_train, ids = _separable_problem(signal=3.0)
    gap = bootstrap_auc_gap(informative, informative, labels, is_train, ids, draws=200)
    assert gap["gap"] == pytest.approx(0.0)
    assert gap["lo"] == pytest.approx(0.0)
    assert gap["hi"] == pytest.approx(0.0)


def test_a_balanced_alteration_leaves_the_obstacle_count_unchanged(instance) -> None:
    """The whole point of the balanced condition: hide the count.

    If the obstacle count moved, the condition would not isolate the control's
    reliance on counting obstacles from the effect of flipping another cell.
    """
    rng = np.random.default_rng(0)
    cells = paired_alteration(instance, (16, 16), rng, polarity="free")
    assert cells is not None and len(cells) == 2
    altered = flip_cells(instance.map_design, cells)
    assert (altered <= 0).sum() == (instance.map_design <= 0).sum()


def test_the_matched_control_adds_two_obstacles(instance) -> None:
    """The comparison condition must differ only in the second flip's direction."""
    rng = np.random.default_rng(0)
    cells = paired_alteration(instance, (16, 16), rng, polarity="block")
    assert cells is not None and len(cells) == 2
    altered = flip_cells(instance.map_design, cells)
    assert (altered <= 0).sum() == (instance.map_design <= 0).sum() + 2


def test_the_second_flip_lands_outside_the_first_receptive_field(instance) -> None:
    """Otherwise the two flips share an encoder patch and could cancel out."""
    rng = np.random.default_rng(1)
    for polarity in ("free", "block"):
        cells = paired_alteration(instance, (16, 16), rng, polarity=polarity)
        assert cells is not None
        (first, second) = cells
        assert max(abs(first[0] - second[0]), abs(first[1] - second[1])) > 5


def test_a_paired_alteration_keeps_the_goal_reachable(instance) -> None:
    """The project-wide constraint: an unreachable goal is a broken map, not an attack."""
    from src.grid import is_reachable

    rng = np.random.default_rng(2)
    for polarity in ("free", "block"):
        cells = paired_alteration(instance, (16, 16), rng, polarity=polarity)
        assert cells is not None
        altered = flip_cells(instance.map_design, cells)
        assert is_reachable(altered, instance.start_rc, instance.goal_rc)


def test_a_paired_alteration_never_touches_the_start_or_goal(instance) -> None:
    """Flipping an endpoint changes the problem rather than the map."""
    rng = np.random.default_rng(3)
    for _ in range(10):
        cells = paired_alteration(instance, (16, 16), rng, polarity="block")
        assert cells is not None
        assert instance.start_rc not in cells and instance.goal_rc not in cells


def test_paired_alteration_rejects_an_unknown_polarity(instance) -> None:
    """A typo here would silently produce the wrong experimental condition."""
    with pytest.raises(ValueError, match="polarity"):
        paired_alteration(instance, (16, 16), np.random.default_rng(0), polarity="flip")


def test_evaluate_detector_uses_a_supplied_head_instead_of_refitting() -> None:
    """The transfer test depends on this: the head must not be refitted.

    If `evaluate_detector` quietly refitted, the transfer test would silently
    become eight independent per-condition fits and would answer a different
    question.
    """
    informative, _, labels, is_train, _ = _separable_problem(signal=3.0)
    # A head fitted with the labels inverted, so its coefficient has the wrong
    # sign. Honouring it must send the AUC *below* chance; refitting on this
    # condition's own training half would send it well above.
    wrong_way = fit_detector(informative[is_train], 1 - labels[is_train])

    honoured = evaluate_detector(
        informative[is_train],
        labels[is_train],
        informative[~is_train],
        labels[~is_train],
        model=wrong_way,
    )
    refitted = evaluate_detector(
        informative[is_train], labels[is_train], informative[~is_train], labels[~is_train]
    )
    assert honoured["auc"] < 0.2
    assert refitted["auc"] > 0.8


def test_the_bootstrap_uses_supplied_heads() -> None:
    """Same requirement, for the interval that accompanies every reported gap."""
    informative, noise, labels, is_train, ids = _separable_problem(signal=3.0)
    heads = (
        fit_detector(informative[is_train], labels[is_train]),
        fit_detector(noise[is_train], labels[is_train]),
    )
    # Pass the feature sets swapped: with the heads honoured, the first set is
    # still scored by the informative head, so the gap stays positive.
    gap = bootstrap_auc_gap(
        informative, noise, labels, is_train, ids, draws=300, models=heads
    )
    assert gap["gap"] > 0.2


def test_fitting_the_baseline_on_the_maps_it_scores_inflates_it_to_perfect() -> None:
    """Documents the leak that produced a flat AUC of 1.000, so nobody repeats it.

    The out-of-distribution baseline estimates its clean distribution from a
    set of clean maps. If that set *is* the set being scored, every clean map
    sits inside the fitted distribution by construction, and the score
    separates perfectly. The effect is severe in exactly this project's regime,
    where a pooled layer has more channels (up to 256) than there are clean
    maps to fit on (50), so the fitted covariance passes through the training
    points. The transfer test must therefore inherit the training half's clean
    maps, and `scripts/e4_detector.py` asserts that it does.
    """
    rng = np.random.default_rng(0)
    # 20 vectors in 40 dimensions: fewer examples than channels, as in § 4.2.
    held_out = rng.normal(0.0, 1.0, (20, 40))   # the clean maps being scored
    reference = rng.normal(0.0, 1.0, (20, 40))  # a disjoint set of clean maps
    vectors = np.vstack([held_out, held_out + 0.3])
    labels = np.concatenate([np.zeros(20), np.ones(20)])

    leaked = evaluate_mahalanobis(held_out, vectors, labels)["auc"]
    honest = evaluate_mahalanobis(reference, vectors, labels)["auc"]
    assert leaked > 0.99
    assert honest < leaked
