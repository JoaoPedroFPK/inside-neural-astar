"""PLAN.md step 9: greedy flip search at larger budgets, and mediation vs budget.

Exhaustive search is affordable only at k=1. This script grows damaging flip
sets one cell at a time (`src.perturb.greedy_flip_search`), seeded from the
exhaustive k=1 ranking, for both objectives; then repeats the step-7 swap on
the resulting sets so the mediation split can be plotted against budget.

The random draws from step 3 are the control: greedy is biased toward cells the
planner is sensitive to, and the gap between the two curves is the size of that
bias.

Usage:
    python scripts/greedy_budgets.py --instances 3 --timing-only
    python scripts/greedy_budgets.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.data import load_split  # noqa: E402
from src.perturb import (  # noqa: E402
    Baseline,
    candidate_pool_from_rows,
    greedy_flip_search,
)
from src.planner import load_pretrained_neural_astar  # noqa: E402
from src.swap import run_swap, summarize_swaps  # noqa: E402

BUDGETS = (1, 2, 4, 8)
OBJECTIVE_METRIC = {"optimality": "opt", "efficiency": "exp"}


def load_pools(manifest: pd.DataFrame, objective: str, size: int) -> Dict[int, List]:
    """Build each instance's candidate pool from the exhaustive k=1 rows.

    Args:
        manifest: The step-6 manifest.
        objective: ``"optimality"`` or ``"efficiency"``.
        size: Pool size per instance.

    Returns:
        dict[int, list]: Instance id -> ranked candidate cells.
    """
    exhaustive = manifest[(manifest["method"] == "exhaustive") & (manifest["budget"] == 1)]
    pools: Dict[int, List] = {}
    for instance_id, group in exhaustive.groupby("instance_id"):
        rows = [
            {
                "flip_cells": [tuple(int(v) for v in cell) for cell in row["flip_cells"]],
                "opt_C1O1": row["opt_C1O1"],
                "exp_C1O1": row["exp_C1O1"],
            }
            for _, row in group.iterrows()
        ]
        pools[int(instance_id)] = candidate_pool_from_rows(rows, objective, size)
    return pools


def random_baseline_curve(objective: str) -> List[Dict[str, float]]:
    """Damage per budget from the step-3 random pilot, for comparison.

    Args:
        objective: Which metric to summarise.

    Returns:
        list[dict]: Per budget, the mean metric and the newly-damaged share;
        empty if the pilot's parquet is missing.
    """
    path = config.RESULTS_DIR / "pilot_random.parquet"
    if not path.exists():
        return []
    frame = pd.read_parquet(path)
    metric = "opt_C1O1" if objective == "optimality" else "exp_C1O1"
    baseline = "opt_C0O0" if objective == "optimality" else "exp_C0O0"
    out = []
    for budget in sorted(frame["budget"].unique()):
        group = frame[frame["budget"] == budget]
        was_clean = group[baseline] <= (1.0 if objective == "optimality" else 1.5)
        out.append(
            {
                "budget": int(budget),
                "mean": float(group[metric].mean()),
                "max": float(group[metric].max()),
                "frac_damaged": float(
                    (group.loc[was_clean, metric] > (1.0 if objective == "optimality" else 1.5)).mean()
                ),
            }
        )
    return out


def render_plot(curves: Dict[str, Dict[int, Dict[str, float]]], out_path: Path) -> Path:
    """Plot mediation fraction against budget, one panel per objective.

    Args:
        curves: objective -> budget -> summary from `summarize_swaps`.
        out_path: Where to write the PNG.

    Returns:
        Path: `out_path`.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(curves), figsize=(5.6 * len(curves), 4.4))
    axes = np.atleast_1d(axes)
    for ax, (objective, by_budget) in zip(axes, curves.items()):
        budgets = sorted(by_budget)
        encoder = [by_budget[b]["encoder_median"] for b in budgets]
        search = [by_budget[b]["search_median"] for b in budgets]
        ax.axhline(0, color="0.85", lw=0.8)
        ax.axhline(1, color="0.9", lw=0.8, ls="--")
        ax.plot(budgets, encoder, "o-", label="encoder route")
        ax.plot(budgets, search, "s-", label="search route")
        ax.set_xscale("log", base=2)
        ax.set_xticks(budgets)
        ax.set_xticklabels([str(b) for b in budgets])
        ax.set_xlabel("budget k (cells flipped)")
        ax.set_ylabel("median mediation fraction")
        ax.set_title(f"{objective} objective", fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("E3 — how the mediation split moves with the attack budget", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def main() -> int:
    """Run the greedy search, the swaps, and write the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=int, default=max(BUDGETS))
    parser.add_argument("--pool", type=int, default=64, help="candidates per round")
    parser.add_argument("--instances", type=int, default=None)
    parser.add_argument("--objectives", nargs="+", default=list(OBJECTIVE_METRIC))
    parser.add_argument("--timing-only", action="store_true")
    parser.add_argument("--manifest", type=Path, default=config.RESULTS_DIR / "manifest.parquet")
    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"{args.manifest} not found; run scripts/sweep_k1.py first", file=sys.stderr)
        return 1

    split_name = config.canonical_split(args.split)
    config.set_global_seeds(args.seed)
    split = load_split(split_name)
    planner = load_pretrained_neural_astar()
    manifest = pd.read_parquet(args.manifest)

    instance_ids = sorted(int(i) for i in manifest["instance_id"].unique())
    if args.instances is not None:
        instance_ids = instance_ids[: args.instances]

    all_rows: List[Dict[str, object]] = []
    swap_results: Dict[str, Dict[int, list]] = {o: {b: [] for b in BUDGETS} for o in args.objectives}
    baselines: Dict[int, Baseline] = {}
    started = time.perf_counter()

    for objective in args.objectives:
        pools = load_pools(manifest, objective, args.pool)
        for position, instance_id in enumerate(instance_ids, start=1):
            instance = split.instance(instance_id, seed=args.seed)
            history = greedy_flip_search(
                planner,
                instance,
                budget=args.budget,
                objective=objective,
                candidate_pool=pools[instance_id],
                baselines=baselines,
            )
            all_rows.extend(history)
            for row in history:
                budget = int(row["budget"])
                if budget in BUDGETS:
                    cells = [tuple(int(v) for v in cell) for cell in row["flip_cells"]]
                    swap_results[objective][budget].append(run_swap(planner, instance, cells))
            if position % 10 == 0 or position == len(instance_ids):
                elapsed = time.perf_counter() - started
                print(
                    f"  {objective}: {position}/{len(instance_ids)} maps | "
                    f"{elapsed / 60:.1f} min elapsed",
                    flush=True,
                )

    seconds = time.perf_counter() - started
    print(f"\n{len(all_rows):,} greedy rows in {seconds / 60:.1f} min")

    if args.timing_only:
        scale = len(sorted(int(i) for i in manifest["instance_id"].unique())) / max(
            len(instance_ids), 1
        )
        print(f"extrapolation to all maps: {seconds * scale / 60:.0f} min")
        return 0

    curves = {
        objective: {
            budget: summarize_swaps(results, OBJECTIVE_METRIC[objective])
            for budget, results in by_budget.items()
            if results
        }
        for objective, by_budget in swap_results.items()
    }
    figure = render_plot(curves, config.RESULTS_DIR / "e3_budget.png")

    greedy_frame = pd.DataFrame(all_rows)
    greedy_frame["flip_cells"] = greedy_frame["flip_cells"].apply(
        lambda cells: [list(rc) for rc in cells]
    )
    for column in ("start_rc", "goal_rc"):
        greedy_frame[column] = greedy_frame[column].apply(list)
    combined = pd.concat([manifest, greedy_frame], ignore_index=True)
    combined.to_parquet(args.manifest, index=False)

    report = {
        "meta": {
            "split": split_name,
            "seed": args.seed,
            "pool": args.pool,
            "n_instances": len(instance_ids),
            "minutes": seconds / 60,
        },
        "curves": curves,
        "greedy_damage": {
            objective: [
                {
                    "budget": b,
                    "mean": float(
                        greedy_frame[
                            (greedy_frame["objective"] == objective)
                            & (greedy_frame["budget"] == b)
                        ]["opt_C1O1" if objective == "optimality" else "exp_C1O1"].mean()
                    ),
                    "max": float(
                        greedy_frame[
                            (greedy_frame["objective"] == objective)
                            & (greedy_frame["budget"] == b)
                        ]["opt_C1O1" if objective == "optimality" else "exp_C1O1"].max()
                    ),
                }
                for b in BUDGETS
            ]
            for objective in args.objectives
        },
        "random_baseline": {o: random_baseline_curve(o) for o in args.objectives},
    }
    (config.RESULTS_DIR / "e3_budget.json").write_text(
        json.dumps(report, indent=2, default=float) + "\n"
    )
    (config.RESULTS_DIR / "e3_budget.md").write_text(render_markdown(report, figure) + "\n")
    print(f"wrote e3_budget.md, e3_budget.json, {figure.name}; manifest now "
          f"{len(combined):,} rows")
    return 0


def render_markdown(report: Dict[str, object], figure: Path) -> str:
    """Render the step-9 report.

    Args:
        report: Assembled results.
        figure: Path to the plot.

    Returns:
        str: Markdown document.
    """
    meta = report["meta"]  # type: ignore[index]
    lines = [
        "# Step 9 — larger budgets: greedy search and mediation vs budget",
        "",
        f"Split `{meta['split']}`, seed {meta['seed']}, {meta['n_instances']} maps, "
        f"candidate pool {meta['pool']} cells per round, {meta['minutes']:.0f} min.",
        "",
        "Greedy search adds one cell at a time, re-evaluating every candidate against "
        "the set already chosen. Two approximations, both deliberate: it cannot find "
        "a pair of cells that only hurts jointly, and its candidate pool is the top of "
        "the exhaustive k=1 ranking rather than the whole grid. The random draws of "
        "step 3 are the control for both.",
        "",
    ]
    for objective, rows in report["greedy_damage"].items():  # type: ignore[index]
        metric = "Opt" if objective == "optimality" else "Exp"
        random_rows = {r["budget"]: r for r in report["random_baseline"][objective]}  # type: ignore[index]
        lines += [
            f"## {objective.capitalize()} objective",
            "",
            f"| budget | greedy mean {metric} | greedy worst {metric} | random mean {metric} |",
            "|--------|---------------------|-----------------------|----------------------|",
        ]
        for row in rows:
            random_mean = random_rows.get(row["budget"], {}).get("mean")
            random_text = f"{random_mean:.4f}" if random_mean is not None else "—"
            lines.append(
                f"| {row['budget']} | {row['mean']:.4f} | {row['max']:.4f} | {random_text} |"
            )
        lines.append("")
        curve = report["curves"].get(objective, {})  # type: ignore[index]
        if curve:
            lines += [
                "| budget | encoder route (median) | search route (median) | interaction share |",
                "|--------|------------------------|-----------------------|-------------------|",
            ]
            for budget in sorted(curve):
                summary = curve[budget]
                labels = summary["labels"]
                interaction = labels.get("interaction", 0) / max(summary["n"], 1)
                lines.append(
                    f"| {budget} | {summary['encoder_median']:+.2f} | "
                    f"{summary['search_median']:+.2f} | {interaction:.0%} |"
                )
            lines.append("")
    lines += [f"Plot: `{figure.name}`.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
