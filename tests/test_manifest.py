"""The results manifest keeps the schema every later step queries.

From step 6 onward every evaluated alteration is appended to one table. Later
steps *query* it rather than recomputing, so a renamed or dropped column breaks
work that was finished weeks earlier. The project rule is to add columns, never
to rename them, and this test is what enforces it.

The manifest is a build artifact (~2 MB, not in version control), so these
tests skip when it has not been built.
"""

from __future__ import annotations

import pytest

from src import config

MANIFEST = config.RESULTS_DIR / "manifest.parquet"
needs_manifest = pytest.mark.skipif(
    not MANIFEST.exists(), reason="manifest not built; run `make sweep`"
)

#: The columns the project's schema requires. Adding to this list is fine;
#: removing or renaming one is the thing being guarded against.
REQUIRED = (
    "instance_id",
    "split",
    "seed",
    "start_rc",
    "goal_rc",
    "budget",
    "flip_cells",
    "method",
    "objective",
    "reachable",
    "opt_C0O0",
    "opt_C0O1",
    "opt_C1O0",
    "opt_C1O1",
    "exp_C0O0",
    "exp_C0O1",
    "exp_C1O0",
    "exp_C1O1",
    "classical_opt_cost_altered",
    "classical_expansions_altered",
    "dcost_l1",
    "dcost_frac_under_flips",
)


@pytest.fixture(scope="module")
def manifest():
    """The results manifest as a dataframe."""
    import pandas as pd

    return pd.read_parquet(MANIFEST)


@needs_manifest
def test_every_required_column_is_present(manifest) -> None:
    """The schema is complete."""
    missing = [column for column in REQUIRED if column not in manifest.columns]
    assert not missing, f"manifest is missing {missing}"


@needs_manifest
def test_rows_are_admissible_alterations(manifest) -> None:
    """Every stored alteration kept the goal reachable and names its method."""
    assert manifest["reachable"].all(), "an unreachable alteration was stored"
    assert set(manifest["method"].unique()) <= {"random", "exhaustive", "greedy", "gradient"}
    assert set(manifest["objective"].unique()) <= {"optimality", "efficiency", "none"}
    assert (manifest["budget"] >= 1).all()
    assert (manifest["flip_cells"].apply(len) == manifest["budget"]).all(), (
        "the flipped-cell list must match the stated budget"
    )


@needs_manifest
def test_metrics_are_in_their_valid_ranges(manifest) -> None:
    """Ratios cannot beat the optimum, and expansion counts are positive."""
    assert (manifest["opt_C1O1"] >= 1.0 - 1e-9).all(), "a planner beat the certified optimum"
    assert (manifest["opt_C0O0"] >= 1.0 - 1e-9).all()
    assert (manifest["exp_C1O1"] > 0).all()
    assert (manifest["classical_opt_cost_altered"] > 0).all()


@needs_manifest
def test_one_winner_per_objective_per_group(manifest) -> None:
    """Step 6 tags exactly one worst alteration per (instance, budget, method).

    The efficiency winner may be absent for a group where no alteration met
    the optimality cap, so that one is checked as "at most one".
    """
    exhaustive = manifest[manifest["method"] == "exhaustive"]
    if "is_worst_optimality" not in exhaustive.columns or exhaustive.empty:
        pytest.skip("sweep tags not present")
    grouped = exhaustive.groupby(["instance_id", "budget"])
    assert (grouped["is_worst_optimality"].sum() == 1).all()
    assert (grouped["is_worst_efficiency"].sum() <= 1).all()


@needs_manifest
def test_swap_columns_are_filled_for_the_tagged_rows(manifest) -> None:
    """Step 7's decomposition is stored on the rows it was computed for."""
    if "mediation_opt_encoder" not in manifest.columns:
        pytest.skip("swap columns not present; run `make swap`")
    tagged = manifest[manifest["is_worst_optimality"].fillna(False).astype(bool)]
    assert len(tagged) > 0
    for column in ("opt_C1O0", "opt_C0O1", "exp_C1O0", "exp_C0O1", "optimum_changed"):
        assert tagged[column].notna().all(), f"{column} missing on tagged rows"
    # The locality bound must hold wherever it was measured.
    measured = manifest["dcost_frac_outside_receptive_field"].dropna()
    assert (measured == 0).all(), "a cost-map change escaped the receptive field"


@needs_manifest
def test_feature_join_is_aligned() -> None:
    """The manifest-to-feature join indexes the right cell of the right map.

    `scripts/analyse_damaging_cells.py` attaches a cell's features to the
    alteration that flipped it, by indexing the archive with
    (instance position, row, column). An off-by-one in any of the three would
    produce a plausible-looking table of nonsense, so a sample of rows is
    re-derived from scratch and compared.
    """
    import numpy as np

    from scripts.analyse_damaging_cells import load_joined
    from src.data import load_split
    from src.features import on_any_optimal_path, wall_zone

    archive = config.RESULTS_DIR / "features_test.npz"
    if not archive.exists():
        pytest.skip("feature archive not built; run `make features`")

    frame = load_joined(MANIFEST, "test")
    split = load_split("test")
    rng = np.random.default_rng(0)
    for position in rng.choice(len(frame), 10, replace=False):
        row = frame.iloc[int(position)]
        instance = split.instance(int(row["instance_id"]), seed=0)
        cell = (int(row["cell_row"]), int(row["cell_col"]))
        assert wall_zone(instance.map_design)[cell] == row["wall_zone"]
        indicator, _, _ = on_any_optimal_path(
            instance.map_design, instance.start_rc, instance.goal_rc, instance.optimal_cost
        )
        assert indicator[cell] == row["on_any_optimal_path"]
        assert (instance.map_design[cell] == 1) == (row["is_free"] == 1)


@needs_manifest
def test_single_cell_analyses_filter_out_multi_cell_rows(manifest) -> None:
    """Analyses that explain one flipped cell must not read multi-cell rows.

    The manifest is shared: step 6 writes single-cell alterations and step 9
    appends greedy sets of up to eight cells. Any analysis that takes
    `flip_cells[0]` has to filter first, or it silently attributes an effect of
    eight changes to one of them. That happened once, in the attribution
    curation, and inverted a reported result.
    """
    from scripts.analyse_damaging_cells import load_joined
    from scripts.attribution_study import curate

    import pandas as pd

    from src.data import load_split

    if not (manifest["budget"] > 1).any():
        pytest.skip("no multi-cell rows present; run `make budgets`")

    # The damaging-cell join restricts to single-cell exhaustive rows.
    joined = load_joined(MANIFEST, "test")
    assert (joined["budget"] == 1).all()
    assert (joined["method"] == "exhaustive").all()

    # So does the attribution curation.
    split = load_split("test")
    entries = curate(split, pd.read_parquet(MANIFEST), per_group=3, seed=0)
    for entry in entries:
        if entry["group"] not in ("attacked", "clean"):
            continue
        cell = tuple(entry["cell"])
        matches = manifest[
            (manifest["instance_id"] == entry["instance_id"])
            & (manifest["budget"] == 1)
            & (manifest["method"] == "exhaustive")
            & manifest["flip_cells"].apply(lambda cells: tuple(int(v) for v in cells[0]) == cell)
        ]
        assert len(matches) == 1, (
            f"curated cell {cell} does not correspond to a single-cell alteration"
        )
