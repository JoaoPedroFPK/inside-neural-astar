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
