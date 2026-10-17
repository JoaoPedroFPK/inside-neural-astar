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
    return out


def main() -> int:
    """Run every check and return a non-zero code if any quoted figure is stale."""
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
