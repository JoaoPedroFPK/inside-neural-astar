"""Worst-case probe for the step-3 budget decision (and step-6 timing).

The random pilot measures the *average* flip. The decision PLAN.md step 3 asks
for — which budgets deserve an exhaustive or greedy search — needs the
*worst* flip instead: a planner that shrugs off a random flip may still have a
single cell per map that breaks it.

This script enumerates every admissible k=1 flip on the first N instances
(~1 000 per map), evaluates all of them, and reports per instance the worst
``Opt`` and the worst ``Exp`` found, plus how many cells cause damage at all.
It also prints the measured cost per alteration, which is the extrapolation
the project's working rules require before launching the full step-6 sweep.

Usage:
    python scripts/probe_exhaustive_k1.py --instances 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.data import load_split  # noqa: E402
from src.perturb import (  # noqa: E402
    AlterationRequest,
    Baseline,
    enumerate_k1_flips,
    evaluate_alterations,
)
from src.planner import load_pretrained_neural_astar  # noqa: E402

EXP_THRESHOLD = 1.5
#: Cap on the Opt ratio an "efficiency" flip may cost, per the working rules, step 6
#: ("max Exp-ratio with Opt-ratio <= 1.05").
OPT_CAP_FOR_EFFICIENCY = 1.05


def main() -> int:
    """Run the probe and write `results/probe_k1.json`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--instances", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--out", type=Path, default=config.RESULTS_DIR / "probe_k1.json"
    )
    args = parser.parse_args()

    split_name = config.canonical_split(args.split)
    config.set_global_seeds(args.seed)
    split = load_split(split_name)
    instances = split.instances(seed=args.seed)[: args.instances]
    planner = load_pretrained_neural_astar()

    per_instance: List[Dict[str, object]] = []
    total_rows = 0
    started = time.perf_counter()
    for instance in instances:
        flips = enumerate_k1_flips(instance)
        n_candidates = (config.MAP_SIZE**2) - 2
        requests = [
            AlterationRequest(
                instance=instance, flip_cells=cells, method="exhaustive", objective="none"
            )
            for cells in flips
        ]
        baselines: Dict[int, Baseline] = {}
        rows = evaluate_alterations(
            planner, requests, baselines=baselines, batch_size=args.batch_size, seed=args.seed
        )
        total_rows += len(rows)

        opt = np.array([r["opt_C1O1"] for r in rows])
        exp = np.array([r["exp_C1O1"] for r in rows])
        baseline = baselines[instance.instance_id]
        efficiency_mask = opt <= OPT_CAP_FOR_EFFICIENCY
        worst_opt_idx = int(np.argmax(opt))
        worst_exp_idx = (
            int(np.flatnonzero(efficiency_mask)[np.argmax(exp[efficiency_mask])])
            if efficiency_mask.any()
            else -1
        )
        per_instance.append(
            {
                "instance_id": instance.instance_id,
                "start_rc": list(instance.start_rc),
                "goal_rc": list(instance.goal_rc),
                "n_admissible": len(rows),
                "n_rejected_unreachable": n_candidates - len(rows),
                "baseline_opt": baseline.opt_ratio,
                "baseline_exp": baseline.exp_ratio,
                "worst_opt": float(opt.max()),
                "worst_opt_cell": rows[worst_opt_idx]["flip_cells"][0],
                "worst_exp_within_opt_cap": float(exp[worst_exp_idx]) if worst_exp_idx >= 0 else None,
                "worst_exp_cell": rows[worst_exp_idx]["flip_cells"][0] if worst_exp_idx >= 0 else None,
                "frac_cells_opt_above_1": float(np.mean(opt > 1.0)),
                "frac_cells_newly_suboptimal": (
                    float(np.mean(opt > 1.0)) if baseline.opt_ratio <= 1.0 else None
                ),
                "frac_cells_exp_above_threshold": float(np.mean(exp > EXP_THRESHOLD)),
                "mean_opt": float(opt.mean()),
                "mean_exp": float(exp.mean()),
            }
        )
    seconds = time.perf_counter() - started

    report = {
        "split": split_name,
        "seed": args.seed,
        "n_instances": len(instances),
        "n_alterations": total_rows,
        "seconds": round(seconds, 2),
        "ms_per_alteration": round(1000 * seconds / max(total_rows, 1), 2),
        "exp_threshold": EXP_THRESHOLD,
        "opt_cap_for_efficiency": OPT_CAP_FOR_EFFICIENCY,
        "per_instance": per_instance,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    print(
        f"\nexhaustive k=1 probe — {len(instances)} instances, {total_rows} alterations, "
        f"{seconds:.1f} s ({report['ms_per_alteration']} ms each)\n"
    )
    print("| inst | admissible | base Opt | worst Opt | cell | cells w/ Opt>1 | worst Exp (Opt<=1.05) | cells w/ Exp>1.5 |")
    print("|------|-----------|----------|-----------|------|----------------|----------------------|------------------|")
    for row in per_instance:
        worst_exp = row["worst_exp_within_opt_cap"]
        print(
            f"| {row['instance_id']} | {row['n_admissible']} | {row['baseline_opt']:.3f} | "
            f"{row['worst_opt']:.3f} | {tuple(row['worst_opt_cell'])} | "
            f"{row['frac_cells_opt_above_1']:.1%} | "
            f"{worst_exp:.3f} | {row['frac_cells_exp_above_threshold']:.1%} |"
        )
    full_split_minutes = (
        report["ms_per_alteration"] * (config.MAP_SIZE**2 - 2) * config.SPLIT_SIZES["test"]
        / 1000
        / 60
    )
    print(
        f"\nextrapolation for PLAN.md step 6 (exhaustive k=1, all "
        f"{config.SPLIT_SIZES['test']} test maps): {full_split_minutes:.0f} min "
        f"single-process"
    )
    print(f"wrote {args.out.relative_to(config.REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
