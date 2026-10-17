"""PLAN.md step 2 validation gate.

Checks the two done-criteria of step 2 on real data, prints a report and exits
non-zero if anything fails:

1. **Solver correctness** — the solver's geodesic distances equal the dataset's
   stored distances on every unaltered map of every split, and point-to-point
   Dijkstra agrees with that field at the fixed starts.
2. **Metrics match step 0** — the project's `Opt`/`Exp` metrics, computed
   against the solver's certified optimum and classical-A\\* expansion count,
   reproduce the E0 numbers in `results/e0_reproduce.json`.

It also quantifies how faithfully the numpy solver replicates the upstream
search module, for both the uniform cost map (classical A\\*) and the learned
cost map (Neural A\\*) — the agreement that lets later steps run searches in
numpy and trust the comparison.

Usage:
    python scripts/validate_step2.py [--split test] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.data import load_split  # noqa: E402
from src.exact_solver import distance_field, solve_astar, solve_dijkstra  # noqa: E402
from src.metrics import n_expansions, path_cost, project_metrics  # noqa: E402
from src.planner import (  # noqa: E402
    encode_cost_map,
    load_pretrained_neural_astar,
    make_vanilla_astar,
)


def check_distance_fields(splits: List[str]) -> Dict[str, object]:
    """Compare solver distance fields against the dataset, split by split.

    Args:
        splits: Split names to check.

    Returns:
        dict: ``{"maps": int, "cells": int, "mismatches": list[str], "ok": bool}``.
    """
    mismatches: List[str] = []
    n_maps = n_cells = 0
    for split_name in splits:
        split = load_split(split_name)
        for i in range(len(split)):
            map_design = split.map_designs[i]
            stored = split.opt_dists[i, 0]
            field = distance_field(map_design, split.goal_rc(i))
            free = map_design == config.FREE
            both = free & np.isfinite(field) & np.isfinite(stored)
            if not np.array_equal(field[both], stored[both]):
                mismatches.append(f"{split_name}[{i}]: distances differ")
            elif not np.array_equal(np.isfinite(field) & free, np.isfinite(stored) & free):
                mismatches.append(f"{split_name}[{i}]: reachability differs")
            n_maps += 1
            n_cells += int(both.sum())
    return {
        "maps": n_maps,
        "cells": n_cells,
        "mismatches": mismatches[:10],
        "n_mismatches": len(mismatches),
        "ok": not mismatches,
    }


def main() -> int:
    """Run every check and return a process exit code (0 = all passed)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", help="split for the metric checks")
    parser.add_argument("--seed", type=int, default=0, help="start-selection seed")
    parser.add_argument(
        "--e0",
        type=Path,
        default=config.RESULTS_DIR / "e0_reproduce.json",
        help="E0 report to compare the metrics against",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=config.RESULTS_DIR / "step2_validation.json",
        help="where to write the JSON report",
    )
    args = parser.parse_args()

    split_name = config.canonical_split(args.split)
    config.set_global_seeds(args.seed)
    report: Dict[str, object] = {"split": split_name, "seed": args.seed, "checks": {}}
    checks: Dict[str, object] = report["checks"]  # type: ignore[assignment]

    # --- 1. solver vs dataset, all 1000 maps ------------------------------- #
    started = time.perf_counter()
    fields = check_distance_fields(["train", "valid", "test"])
    fields["seconds"] = round(time.perf_counter() - started, 2)
    checks["distance_fields_vs_dataset"] = fields

    # --- 2. planners and solver on the chosen split ------------------------ #
    split = load_split(split_name)
    instances = split.instances(seed=args.seed)
    tensors = [inst.to_tensors() for inst in instances]
    maps = torch.cat([t[0] for t in tensors])
    starts = torch.cat([t[1] for t in tensors])
    goals = torch.cat([t[2] for t in tensors])

    planner = load_pretrained_neural_astar()
    vanilla = make_vanilla_astar()
    with torch.no_grad():
        neural_out = planner(maps, starts, goals)
        vanilla_out = vanilla(maps, starts, goals)
    cost_maps = encode_cost_map(planner, maps, starts, goals).numpy()[:, 0]

    started = time.perf_counter()
    optimum, classical_exp, solver_classical_cost = [], [], []
    solver_neural_exp, solver_neural_moves = [], []
    dataset_optimum = []
    for k, inst in enumerate(instances):
        dijkstra = solve_dijkstra(inst.map_design, inst.start_rc, inst.goal_rc)
        classical = solve_astar(inst.map_design, inst.start_rc, inst.goal_rc)
        weighted = solve_astar(
            inst.map_design,
            inst.start_rc,
            inst.goal_rc,
            cost_map=cost_maps[k].astype(np.float64),
        )
        optimum.append(dijkstra.cost)
        classical_exp.append(classical.expansions)
        solver_classical_cost.append(classical.cost)
        solver_neural_exp.append(weighted.expansions)
        solver_neural_moves.append(weighted.n_moves)
        dataset_optimum.append(inst.optimal_cost)
    solver_seconds = round(time.perf_counter() - started, 2)

    optimum = np.array(optimum)
    classical_exp = np.array(classical_exp, dtype=np.float64)
    checks["dijkstra_vs_dataset_at_fixed_starts"] = {
        "ok": bool(np.array_equal(optimum, np.array(dataset_optimum))),
        "n": len(instances),
    }

    # --- 3. solver vs the upstream search module --------------------------- #
    checks["solver_replicates_upstream"] = {
        "classical_expansions_match": float(
            (classical_exp == n_expansions(vanilla_out.histories)).mean()
        ),
        "classical_costs_match": float(
            (np.array(solver_classical_cost) == path_cost(vanilla_out.paths)).mean()
        ),
        "neural_expansions_match": float(
            (np.array(solver_neural_exp) == n_expansions(neural_out.histories)).mean()
        ),
        "neural_path_moves_match": float(
            (np.array(solver_neural_moves) == path_cost(neural_out.paths)).mean()
        ),
        "seconds_for_3_solves_per_instance": solver_seconds,
    }
    checks["solver_replicates_upstream"]["ok"] = all(  # type: ignore[index]
        value == 1.0
        for key, value in checks["solver_replicates_upstream"].items()  # type: ignore[union-attr]
        if key.endswith("_match")
    )

    # --- 4. project metrics, and agreement with E0 ------------------------- #
    metrics = project_metrics(neural_out, optimum, classical_exp)
    summary = metrics["summary"]
    checks["project_metrics"] = summary

    e0_match: Dict[str, object] = {"available": args.e0.exists()}
    if args.e0.exists():
        e0 = json.loads(args.e0.read_text())
        protocol = e0["protocols"].get("fixed")
        if protocol is not None:
            e0_match.update(
                {
                    "e0_p_opt": protocol["p_opt"],
                    "solver_p_opt": summary["p_opt"],  # type: ignore[index]
                    "p_opt_ok": abs(protocol["p_opt"] - summary["p_opt"]) < 1e-9,  # type: ignore[index]
                    "e0_p_exp": protocol["p_exp"],
                    "solver_p_exp": summary["p_exp"],  # type: ignore[index]
                    "p_exp_ok": abs(protocol["p_exp"] - summary["p_exp"]) < 1e-9,  # type: ignore[index]
                }
            )
            e0_match["ok"] = bool(e0_match["p_opt_ok"] and e0_match["p_exp_ok"])
    checks["metrics_match_e0"] = e0_match

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=float) + "\n")

    # --- report ------------------------------------------------------------ #
    print(f"\nPLAN.md step 2 validation — split={split_name}, seed={args.seed}\n")
    print(
        f"1. distance field vs dataset: {fields['maps']} maps, "
        f"{fields['cells']:,} free cells, {fields['n_mismatches']} mismatches "
        f"({fields['seconds']} s)"
    )
    print(
        f"2. point-to-point Dijkstra vs dataset at fixed starts: "
        f"{'ok' if checks['dijkstra_vs_dataset_at_fixed_starts']['ok'] else 'FAILED'}"  # type: ignore[index]
    )
    rep = checks["solver_replicates_upstream"]
    print("3. numpy solver vs upstream differentiable A* module:")
    print(f"     classical A*  expansions match: {rep['classical_expansions_match']:.0%}")  # type: ignore[index]
    print(f"     classical A*  path costs  match: {rep['classical_costs_match']:.0%}")  # type: ignore[index]
    print(f"     Neural A*     expansions match: {rep['neural_expansions_match']:.0%}")  # type: ignore[index]
    print(f"     Neural A*     path moves  match: {rep['neural_path_moves_match']:.0%}")  # type: ignore[index]
    print("4. project metrics on the pretrained model:")
    print(
        f"     Opt  mean ratio {summary['mean_opt_ratio']:.4f}   "  # type: ignore[index]
        f"binary {summary['p_opt']:.3f}"  # type: ignore[index]
    )
    print(
        f"     Exp  mean ratio {summary['mean_exp_ratio']:.4f}   "  # type: ignore[index]
        f"worse than classical on {summary['n_worse_than_classical']}/{summary['n']} instances"  # type: ignore[index]
    )
    print(
        f"     paper-style p_opt {summary['p_opt']:.3f}  p_exp {summary['p_exp']:.4f}  "  # type: ignore[index]
        f"h_mean {summary['h_mean']:.4f}"  # type: ignore[index]
    )
    if e0_match.get("available"):
        print(
            f"5. matches E0 (fixed protocol): p_opt {e0_match['e0_p_opt']:.3f} vs "
            f"{e0_match['solver_p_opt']:.3f}, p_exp {e0_match['e0_p_exp']:.4f} vs "
            f"{e0_match['solver_p_exp']:.4f} -> {'ok' if e0_match.get('ok') else 'FAILED'}"
        )
    else:
        print("5. E0 report not found; run scripts/e0_reproduce.py first")

    failed = [
        name
        for name, check in checks.items()
        if isinstance(check, dict) and check.get("ok") is False
    ]
    print(f"\nwrote {args.out.relative_to(config.REPO_ROOT)}")
    if failed:
        print(f"FAILED checks: {', '.join(failed)}")
        return 1
    print("all step-2 checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
