"""PLAN.md step 5: surrogate models of the cost map.

Unit tests run on synthetic design matrices so the expected answer is known.
The tests that need the real feature archives skip when they are absent, since
the archives are build artifacts (`make features`) rather than repository
content.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import config
from src.features import FEATURE_NAMES
from src.surrogate import (
    FEATURE_GROUPS,
    DesignMatrix,
    build_design_matrix,
    fit_and_score,
    load_archive,
    make_model,
    residual_by_class,
    sufficiency_cost_maps,
    within_instance_r2,
)

ARCHIVES_PRESENT = (config.RESULTS_DIR / "features_test.npz").exists() and (
    config.RESULTS_DIR / "features_train.npz"
).exists()
needs_archives = pytest.mark.skipif(
    not ARCHIVES_PRESENT, reason="feature archives not built; run `make features`"
)


def _synthetic(n: int = 400, seed: int = 0) -> DesignMatrix:
    """A design matrix whose target is an exact linear function of column 0."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(FEATURE_NAMES)))
    y = 0.3 * X[:, 0] + 0.5
    return DesignMatrix(
        X=X,
        y=y,
        feature_names=tuple(FEATURE_NAMES),
        instance_index=np.repeat(np.arange(n // 100), 100),
        wall_zone=np.tile(np.array([0, 1, 2, 3], dtype=np.int8), n // 4),
        topology_class=np.zeros(n, dtype=np.int8),
        split="test",
    )


# --------------------------------------------------------------------------- #
# Feature groups
# --------------------------------------------------------------------------- #

def test_groups_partition_the_feature_library() -> None:
    """`geometric` and `path` together are exactly the whole library."""
    assert set(FEATURE_GROUPS["geometric"]) | set(FEATURE_GROUPS["path"]) == set(FEATURE_NAMES)
    assert not set(FEATURE_GROUPS["geometric"]) & set(FEATURE_GROUPS["path"])
    assert set(FEATURE_GROUPS["all"]) == set(FEATURE_NAMES)


def test_h1_group_excludes_the_leaking_feature() -> None:
    """`geometric_h1` must not contain `geodesic_from_start`.

    A cell is on some optimal path exactly when the two geodesic distances sum
    to the optimum, so including the start distance hands a non-linear model
    the path indicator for free - which would make `geometric` a test of the
    wrong hypothesis.
    """
    assert "geodesic_from_start" not in FEATURE_GROUPS["geometric_h1"]
    assert "geodesic_from_start" in FEATURE_GROUPS["geometric"]
    assert set(FEATURE_GROUPS["geometric_h1"]) < set(FEATURE_GROUPS["geometric"])
    assert not any(n.startswith("on_") for n in FEATURE_GROUPS["geometric_h1"])


def test_heuristic_group_spans_the_search_heuristic() -> None:
    """The two features that reconstruct upstream's `h` are grouped together."""
    assert set(FEATURE_GROUPS["heuristic"]) == {"chebyshev_to_goal", "euclidean_to_goal"}


# --------------------------------------------------------------------------- #
# Design matrix
# --------------------------------------------------------------------------- #

def test_subset_selects_the_right_columns() -> None:
    """`subset` keeps the group's columns, in the group's order."""
    design = _synthetic()
    sub = design.subset("path")
    assert sub.feature_names == FEATURE_GROUPS["path"]
    expected = design.X[:, [FEATURE_NAMES.index(n) for n in FEATURE_GROUPS["path"]]]
    assert np.array_equal(sub.X, expected)
    assert np.array_equal(sub.y, design.y)


def test_within_instance_r2_behaviour() -> None:
    """Perfect prediction scores 1; predicting each instance's mean scores 0."""
    y = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
    inst = np.array([0, 0, 0, 1, 1, 1])
    assert abs(within_instance_r2(y, y, inst) - 1.0) < 1e-12
    per_mean = np.array([2.0, 2.0, 2.0, 11.0, 11.0, 11.0])
    assert abs(within_instance_r2(y, per_mean, inst)) < 1e-12


def test_fit_recovers_a_linear_relationship() -> None:
    """Ridge on a column that determines the target scores R2 ~ 1."""
    train, test = _synthetic(seed=0), _synthetic(seed=1)
    result = fit_and_score(train, test, "all", "ridge")
    assert result.r2 > 0.99
    assert result.n_features == len(FEATURE_NAMES)
    assert result.coefficients is not None
    strongest = max(result.coefficients, key=lambda k: abs(result.coefficients[k]))
    assert strongest == FEATURE_NAMES[0]


def test_gbm_also_fits_and_reports_no_coefficients() -> None:
    """The boosting model fits and has no linear coefficients to report."""
    train, test = _synthetic(seed=0), _synthetic(seed=1)
    result = fit_and_score(train, test, "path", "gbm")
    assert result.kind == "gbm"
    assert result.coefficients is None
    assert np.isfinite(result.rmse)


def test_unknown_model_kind_is_rejected() -> None:
    """A typo in the model name fails loudly."""
    with pytest.raises(ValueError, match="unknown model kind"):
        make_model("randomforest")


def test_residual_by_class_covers_every_cell() -> None:
    """Class shares sum to 1 and every class reports finite residuals."""
    train, test = _synthetic(seed=0), _synthetic(seed=1)
    result = fit_and_score(train, test, "all", "ridge")
    table = residual_by_class(result, test, "wall_zone")
    assert abs(sum(row["share"] for row in table) - 1.0) < 1e-9
    assert all(np.isfinite(row["mean_abs_residual"]) for row in table)
    with pytest.raises(ValueError, match="unknown grouping"):
        residual_by_class(result, test, "nonsense")


# --------------------------------------------------------------------------- #
# Against the real archives
# --------------------------------------------------------------------------- #

@needs_archives
def test_design_matrix_rows_are_usable_cells() -> None:
    """Rows are free, reachable cells only, with no infinities left."""
    archive = load_archive("test")
    design = build_design_matrix("test", archive)
    mask = (archive["is_free"] > 0) & (archive["reachable"] > 0)
    assert len(design.y) == int(mask.sum())
    assert np.isfinite(design.X).all(), "an unreachable cell leaked into the matrix"
    assert len(design.instance_index) == len(design.y)
    assert design.instance_index.max() == archive["cost_map"].shape[0] - 1


@needs_archives
def test_sufficiency_cost_maps_are_valid_cost_maps() -> None:
    """Predicted cost maps are map-shaped and inside the sigmoid's range.

    They are fed to the real search module, which assumes costs in (0, 1); a
    value outside that range would change the search's behaviour for reasons
    unrelated to the surrogate's quality.
    """
    archive = load_archive("test")
    train = build_design_matrix("train")
    test = build_design_matrix("test", archive)
    result = fit_and_score(train, test, "geometric_h1", "ridge")
    maps = sufficiency_cost_maps(result, archive)
    assert maps.shape == archive["cost_map"].shape
    assert maps.dtype == np.float32
    assert maps.min() > 0.0 and maps.max() < 1.0


@needs_archives
def test_path_features_beat_the_heuristic_features() -> None:
    """A regression guard on the headline qualitative result.

    The search already supplies the Chebyshev heuristic, so the encoder has no
    reason to re-encode it, and measurement agrees: the heuristic features
    explain almost nothing while the path indicators explain an order of
    magnitude more. Loose bounds - this guards the conclusion, not the digits.
    """
    train, test = build_design_matrix("train"), build_design_matrix("test")
    heuristic = fit_and_score(train, test, "heuristic", "ridge")
    path = fit_and_score(train, test, "path", "ridge")
    assert heuristic.r2 < 0.02
    assert path.r2 > 0.08
    assert path.r2 > 4 * heuristic.r2


@needs_archives
def test_tagged_archives_are_kept_separate() -> None:
    """A model tag selects that model's archive, not the default one.

    Several trained encoders coexist in `results/`, and mixing one model's cost
    maps with another's would silently invalidate every per-seed comparison.
    """
    import numpy as np

    from src import config

    default = load_archive("test")
    default_family = str(default.get("family", "mazes"))

    # Only archives from the *same* family are comparable: the geometry arrays
    # are properties of the maps, so a forest archive differs from a maze one
    # for reasons that have nothing to do with the model.
    same_family = []
    for path in sorted(config.RESULTS_DIR.glob("features_test_*.npz")):
        tag = path.stem.replace("features_test_", "")
        archive = load_archive("test", tag=tag)
        if str(archive.get("family", "mazes")) == default_family:
            same_family.append((tag, archive))
    if not same_family:
        pytest.skip("no per-model archives for this family; run scripts/seed_pipeline.sh")

    tag, other = same_family[0]
    assert str(other["model_tag"]) == tag
    # The geometry is a property of the maps and must be identical...
    assert np.array_equal(default["geodesic_to_goal"], other["geodesic_to_goal"])
    assert np.array_equal(default["is_free"], other["is_free"])
    # ...while the cost map is a property of the model and must not be.
    assert not np.array_equal(default["cost_map"], other["cost_map"])


@needs_archives
def test_archives_from_different_families_are_not_comparable() -> None:
    """An archive records its family, so the two cannot be mixed up silently.

    Per-model archives began as "same maps, different weights". Once step 7b
    trained on another family that stopped being true, and a comparison that
    assumed it would have reported a difference in *geometry* as a difference
    between models.
    """
    import numpy as np

    from src import config

    others = {}
    for path in sorted(config.RESULTS_DIR.glob("features_test_*.npz")):
        tag = path.stem.replace("features_test_", "")
        archive = load_archive("test", tag=tag)
        others.setdefault(str(archive.get("family", "mazes")), []).append((tag, archive))
    if len(others) < 2:
        pytest.skip("only one family has per-model archives")

    (first_family, first), (second_family, second) = (
        (name, entries[0][1]) for name, entries in list(others.items())[:2]
    )
    assert first_family != second_family
    assert not np.array_equal(first["is_free"], second["is_free"]), (
        "two families must differ in their maps, not only in their cost maps"
    )
