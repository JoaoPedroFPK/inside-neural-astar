"""Check that the numbers quoted in the write-ups match the stored results.

The step write-ups and the README quote figures by hand. Results get
regenerated, scripts get improved, and a quoted number can quietly go stale —
which is worse than having no number at all, because it still looks precise.

This script re-reads the stored result files and asserts the headline figures
are what the prose says. It is maintenance tooling: run it before publishing
anything, and after any rerun.

Usage:
    python scripts/verify_reported_numbers.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402


def load(name: str):
    """Load one result file.

    Args:
        name: Path relative to the results directory.

    Returns:
        The parsed JSON, or None when the file has not been produced yet.
    """
    path = config.RESULTS_DIR / name
    if not path.exists():
        return None
    return json.loads(path.read_text())


def checks() -> List[Tuple[str, Callable[[], bool]]]:
    """The quoted figures, each with the comparison that validates it.

    Returns:
        list[tuple[str, Callable]]: Description and a predicate. A check whose
        inputs are missing returns None from its loader and is skipped.
    """
    e0 = load("e0_reproduce.json")
    step2 = load("step2_validation.json")
    surrogate = load("e1_surrogate.json")
    sweep = load("e2_k1_sweep.json")
    swap = load("e3_k1.json")
    damaging = load("e2_damaging_cells.json")
    attribution = load("e1_attribution/summary.json")
    seeds = load("seeds.json")
    detector = load("e4_detector.json")
    transfer = load("e4_detector_from_greedy_opt_b4.json")
    detector_models = [
        load(f"e4_detector_{tag}.json")
        for tag in ("mazes_seed0", "mazes_seed1", "mazes_seed2")
    ]
    seeds_forest = load("seeds_forest.json")
    detector_forest = [
        load(f"e4_detector_{tag}.json") for tag in ("forest_seed0", "forest_seed1")
    ]

    out: List[Tuple[str, Callable[[], bool]]] = []

    def near(value: float, target: float, tolerance: float = 5e-4) -> bool:
        return abs(value - target) <= tolerance

    if e0:
        out += [
            ("E0, upstream protocol: Opt 0.840", lambda: near(e0["protocols"]["upstream"]["p_opt"], 0.840)),
            ("E0, fixed starts: Opt 0.830", lambda: near(e0["protocols"]["fixed"]["p_opt"], 0.830)),
            ("E0, fixed starts: Exp 0.4373", lambda: near(e0["protocols"]["fixed"]["p_exp"], 0.4373)),
        ]
    if step2:
        out.append(
            ("step 2: the solver replicates the upstream module",
             lambda: step2["checks"]["solver_replicates_upstream"]["ok"])
        )
    if surrogate:
        fits = {(f["group"], f["kind"]): f["r2"] for f in surrogate["fits"]}
        out += [
            ("step 5: heuristic features explain 0.002", lambda: near(fits[("heuristic", "ridge")], 0.002)),
            ("step 5: H1 geometry reaches 0.240", lambda: near(fits[("geometric_h1", "gbm")], 0.240)),
            ("step 5: all features reach 0.408", lambda: near(fits[("all", "gbm")], 0.408)),
            ("step 5: surrogate plans 0.920 optimal",
             lambda: near(surrogate["sufficiency"]["surrogate_all"]["p_opt"], 0.920)),
            ("step 5: surrogate Hmean 0.552",
             lambda: near(surrogate["sufficiency"]["surrogate_all"]["h_mean"], 0.552, 1e-3)),
        ]
    if sweep:
        overall = sweep["overall"]
        out += [
            ("step 6: 86% of maps have a damaging cell",
             lambda: near(overall["frac_instances_with_a_damaging_cell"], 0.86, 0.01)),
            ("step 6: median 0.78% of cells are damaging",
             lambda: near(overall["median_frac_cells_damaging"], 0.0078)),
            ("step 6: worst single flip 1.353", lambda: near(overall["max_worst_opt"], 1.353, 1e-3)),
        ]
    if swap:
        unchanged = swap["summaries_by_failure_mode"]["optimality"]["optimum unchanged"]
        out += [
            ("step 7: encoder 0.493 when the optimum is unchanged",
             lambda: near(unchanged["encoder_mean"], 0.493, 2e-3)),
            ("step 7: search 0.284 when the optimum is unchanged",
             lambda: near(unchanged["search_mean"], 0.284, 2e-3)),
            ("step 7: search 0.497 on the efficiency objective",
             lambda: near(swap["summaries"]["efficiency"]["search_mean"], 0.497, 2e-3)),
        ]
    if damaging:
        out.append(("step 6: 2.00% of all flips damage the planner",
                    lambda: near(damaging["overall_rate"], 0.0200)))
    if attribution:
        locality = {r["method"]: r for r in attribution["locality_by_method"]}
        out += [
            ("step 8: Grad-CAM keeps 10.9% of its mass in the field",
             lambda: near(locality["grad_cam"]["share_within_5"], 0.109, 5e-3)),
            # The share is a float32 sum, so it lands a few ulps below 1. The
            # rigorous statement is the radius: no non-zero attribution exists
            # beyond the bound at all.
            ("step 8: integrated gradients stay inside the field",
             lambda: locality["integrated_gradients"]["max_radius"] <= 5.0),
            ("step 8: saliency stays inside the field",
             lambda: locality["saliency"]["max_radius"] <= 5.0),
        ]
    if seeds:
        models = {row["tag"]: row for row in seeds["models"]}
        fragility = [
            row["fragility"]["median_damaging_cells"]
            for row in seeds["models"]
            if "fragility" in row
        ]
        if len(fragility) > 1:
            out.append(
                ("step 10: fragility replicates across models (median damaging cells)",
                 lambda: max(fragility) - min(fragility) < 0.005)
            )
        encoders = [
            row["mediation_optimality"]["encoder"]
            for row in seeds["models"]
            if "mediation_optimality" in row
        ]
        searches = [
            row["mediation_optimality"]["search"]
            for row in seeds["models"]
            if "mediation_optimality" in row
        ]
        efficiency_encoders = [
            row["mediation_efficiency"]["encoder"]
            for row in seeds["models"]
            if "mediation_efficiency" in row
        ]
        efficiency_searches = [
            row["mediation_efficiency"]["search"]
            for row in seeds["models"]
            if "mediation_efficiency" in row
        ]
        if len(encoders) > 1:
            # The step-10 claim is about *ordering*, not magnitude: the two
            # routes' ranges do not overlap, in opposite directions for the two
            # objectives. The magnitudes move by about 0.16 either way, which
            # is why no claim in this project quotes a single one of them.
            out.append(
                ("step 10: the encoder route leads on path quality in every model",
                 lambda: min(encoders) > max(searches))
            )
        if len(efficiency_encoders) > 1:
            out.append(
                ("step 10: the search route leads on search effort in every model",
                 lambda: min(efficiency_searches) > max(efficiency_encoders))
            )
    if detector:
        conditions = {row["condition"]: row for row in detector["conditions"]}
        out.extend(
            [
                ("step 12: the model-free control already reaches 0.779 on single flips",
                 lambda: near(conditions["k1_opt"]["map_only_auc"], 0.779, 1e-3)),
                ("step 12: the residual alone never beats the activation baseline",
                 lambda: all(
                     row["residual_auc"] < row["mahalanobis_auc"]
                     for row in detector["conditions"]
                 )),
                ("step 12: on single flips the residual adds 0.113 over the control",
                 lambda: near(
                     conditions["k1_opt"]["combined_minus_map_only"]["gap"], 0.113, 1e-3
                 )),
                ("step 12: that gain's interval excludes zero",
                 lambda: conditions["k1_opt"]["combined_minus_map_only"]["lo"] > 0),
                ("step 12: at budget 8 on path quality the residual adds nothing",
                 lambda: conditions["greedy_opt_b8"]["combined_minus_map_only"]["lo"] < 0),
                # Holding the number of flips at two, hiding the obstacle count
                # costs the model-free control 0.124 AUC and the activation
                # baseline nothing. Both halves matter: the first shows the
                # control counts obstacles, the second shows the baseline does
                # not, which is what § 4.2 of the write-up had to be corrected to.
                ("step 12: hiding the obstacle count costs the control 0.124 AUC",
                 lambda: near(
                     conditions["blocked_2"]["map_only_auc"]
                     - conditions["balanced_2"]["map_only_auc"],
                     0.124,
                     1e-3,
                 )),
                ("step 12: hiding the obstacle count costs the activation baseline nothing",
                 lambda: abs(
                     conditions["blocked_2"]["mahalanobis_auc"]
                     - conditions["balanced_2"]["mahalanobis_auc"]
                 ) < 0.01),
                # The expectation that the realistic, unpaired task would be
                # harder was written into the step and then measured. It was
                # wrong in the direction that matters, so both halves are
                # pinned: the control loses AUC, the combination does not.
                ("step 12: dropping the clean twin costs the control AUC",
                 lambda: conditions["k1_opt"]["unpaired"]["map_only"]
                 < conditions["k1_opt"]["map_only_auc"]),
                ("step 12: dropping the clean twin does not cost the combined detector",
                 lambda: conditions["k1_opt"]["unpaired"]["combined"]
                 >= conditions["k1_opt"]["combined_auc"]),
                ("step 12: without a clean twin the grid control falls below chance "
                 "on stealthy effort flips",
                 lambda: conditions["greedy_eff_b1"]["unpaired"]["map_only"] < 0.5),
                # The claim the step actually reports: measured against a
                # convolutional map-only control, the residual's contribution
                # survives at one flipped cell and vanishes at two or more.
                ("step 12: against the convolutional control the residual still helps "
                 "at one flipped cell",
                 lambda: all(
                     conditions[name]["grid_cnn"]["gap"]["lo"] > 0
                     for name in ("k1_opt", "k1_eff", "greedy_eff_b1")
                 )),
                ("step 12: against the convolutional control it adds nothing at two cells "
                 "or more",
                 lambda: all(
                     conditions[name]["grid_cnn"]["gap"]["lo"] <= 0
                     for name in (
                         "balanced_2",
                         "blocked_2",
                         "greedy_opt_b4",
                         "greedy_opt_b8",
                         "greedy_eff_b4",
                         "greedy_eff_b8",
                     )
                 )),
                ("step 12: the convolutional control beats the hand-feature detector "
                 "on single path flips",
                 lambda: conditions["k1_opt"]["grid_cnn"]["auc"]
                 > conditions["k1_opt"]["combined_auc"]),
                ("step 12: two added obstacles are perfectly visible in the grid alone",
                 lambda: near(conditions["blocked_2"]["grid_cnn"]["auc"], 1.0, 1e-6)),
                ("step 12: the residual helps more against the stealthier attack",
                 lambda: conditions["balanced_2"]["combined_minus_map_only"]["gap"]
                 > conditions["blocked_2"]["combined_minus_map_only"]["hi"]),
            ]
        )

    if seeds_forest:
        forest_models = seeds_forest["models"]
        forest_path_encoder = [
            row["mediation_optimality"]["encoder"]
            for row in forest_models
            if "mediation_optimality" in row
        ]
        forest_path_search = [
            row["mediation_optimality"]["search"]
            for row in forest_models
            if "mediation_optimality" in row
        ]
        forest_effort_encoder = [
            row["mediation_efficiency"]["encoder"]
            for row in forest_models
            if "mediation_efficiency" in row
        ]
        forest_effort_search = [
            row["mediation_efficiency"]["search"]
            for row in forest_models
            if "mediation_efficiency" in row
        ]
        if len(forest_path_encoder) > 2:
            out.extend(
                [
                    # The half that travels between families.
                    ("step 7b: the encoder route leads on path quality in every "
                     "forest model too",
                     lambda: min(forest_path_encoder) > max(forest_path_search)),
                    # The half that does not: the step-7b hunch was withdrawn
                    # because these ranges overlap, so the check is that they do.
                    ("step 7b: on forest the search-effort route is undetermined "
                     "(ranges overlap)",
                     lambda: min(forest_effort_search) < max(forest_effort_encoder)),
                    ("step 7b: the third forest seed reverses effort by more than any "
                     "maze model's margin",
                     lambda: max(
                         search - encoder
                         for encoder, search in zip(forest_effort_encoder, forest_effort_search)
                     ) > 0.33),
                ]
            )

    if transfer:
        moved = {row["condition"]: row for row in transfer["conditions"]}
        # PLAN.md's design: one detector fitted on k=4 optimality flips, scored
        # across budgets and objectives. The claim is that transfer is nearly
        # free, so the comparison is against the per-condition fits.
        out.extend(
            [
                ("step 12: a transferred detector loses at most 0.05 AUC per condition",
                 lambda: max(
                     conditions[name]["residual_auc"] - moved[name]["residual_auc"]
                     for name in moved
                     if name in conditions
                 ) < 0.05),
                ("step 12: the transferred detector still beats the control on single flips",
                 lambda: moved["k1_opt"]["combined_minus_map_only"]["lo"] > 0),
                ("step 12: the stealth ordering survives transfer",
                 lambda: moved["balanced_2"]["combined_minus_map_only"]["gap"]
                 > moved["blocked_2"]["combined_minus_map_only"]["gap"]),
            ]
        )

    present = [payload for payload in detector_models if payload]
    if len(present) > 1:
        # The step-12 claim that travels between models is the *sign* of the
        # gain, not its size — the same standard step 10 imposed on mediation.
        gains = [
            row["combined_minus_map_only"]["lo"]
            for payload in present
            for row in payload["conditions"]
        ]
        unpaired_gains = [
            row["unpaired_combined_minus_map_only"]["lo"]
            for payload in present
            for row in payload["conditions"]
        ]
        out.extend(
            [
                ("step 12: the residual's gain over the control holds in every maze model",
                 lambda: min(gains) > 0),
                ("step 12: it holds in every maze model on the unpaired task too",
                 lambda: min(unpaired_gains) > 0),
            ]
        )

    forest = [payload for payload in detector_forest if payload]
    if len(forest) > 1:
        # The forest null is the opposite claim and needs the opposite check:
        # every interval must *contain* zero, in every forest model.
        intervals = [
            row["combined_minus_map_only"]
            for payload in forest
            for row in payload["conditions"]
        ]
        unpaired_intervals = [
            row["unpaired_combined_minus_map_only"]
            for payload in forest
            for row in payload["conditions"]
        ]
        out.extend(
            [
                ("step 12: on forest the residual adds nothing, in every model",
                 lambda: all(row["lo"] <= 0 <= row["hi"] for row in intervals)),
                ("step 12: the forest null holds on the unpaired task too",
                 lambda: all(
                     row["lo"] <= 0 <= row["hi"] for row in unpaired_intervals
                 )),
            ]
        )
        # And the baseline fails there too, which is why the reading is about
        # the internals as a whole rather than about the residual.
        baselines = [
            row["mahalanobis_auc"] for payload in forest for row in payload["conditions"]
        ]
        out.append(
            ("step 12: on forest the activation baseline collapses as well",
             lambda: max(baselines) < 0.65)
        )
    return out


def main() -> int:
    """Run every check and return a non-zero code if any quoted figure is stale."""
    import argparse

    # This script is also exercised by `tests/test_repo_hygiene.py`, which runs
    # every script with `--help` and expects exit code 0. Without an argument
    # parser, `--help` ran the checks instead, so one stale figure surfaced as
    # "this script has a broken command line".
    argparse.ArgumentParser(description=__doc__).parse_args()
    results = checks()
    if not results:
        print("no result files found; run the experiments first", file=sys.stderr)
        return 1

    failures = []
    for description, predicate in results:
        try:
            ok = bool(predicate())
        except Exception as error:  # pragma: no cover - a malformed result file
            ok, description = False, f"{description} (raised {error!r})"
        print(("  ok    " if ok else "  STALE ") + description)
        if not ok:
            failures.append(description)

    print(f"\n{len(results) - len(failures)}/{len(results)} quoted figures match the stored results")
    if failures:
        print("stale figures: " + "; ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
