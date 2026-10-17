"""PLAN.md step 6: exhaustive single-cell flip sweep (E2).

For every test map and every candidate cell (~1 022 per map), flips the cell,
checks the goal is still reachable, and evaluates the altered map with Neural
A\\* and with the exact solver. Caches every result in
`results/manifest.parquet` and tags, per map, the worst flip under each
objective:

* **optimality** — the flip maximising the Opt ratio;
* **efficiency** — the flip maximising the Exp ratio among those costing at
  most 5 % of path quality, so the two objectives stay distinct.

Usage:
    python scripts/sweep_k1.py --instances 3 --timing-only   # extrapolate first
    python scripts/sweep_k1.py                               # the full sweep
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.data import load_split  # noqa: E402
from src.perturb import (  # noqa: E402
    OPT_CAP_FOR_EFFICIENCY,
    AlterationRequest,
    Baseline,
    enumerate_k1_flips,
    evaluate_alterations,
    select_worst_flips,
    summarize_sweep,
)
from src.planner import load_pretrained_neural_astar  # noqa: E402


def render_markdown(summary: Dict[str, object], meta: Dict[str, object]) -> str:
    """Render the step-6 report.

    Args:
        summary: Output of `src.perturb.summarize_sweep`.
        meta: Run metadata.

    Returns:
        str: Markdown document.
    """
    overall = summary["overall"]  # type: ignore[index]
    per_instance = summary["per_instance"]  # type: ignore[index]
    worst = sorted(per_instance, key=lambda p: -p["worst_opt"])[:10]
    lines = [
        "# Step 6 — exhaustive single-cell flip sweep",
        "",
        f"Split `{meta['split']}`, seed {meta['seed']}: every admissible single-cell "
        f"flip of every map — {overall['n_alterations']:,} alterations over "
        f"{overall['n_instances']} maps, {meta['minutes']:.1f} min "
        f"({meta['ms_per_alteration']:.1f} ms each).",
        "",
        "Each alteration is scored against the **altered** map's own optimum and "
        "classical-A\\* effort, so a planner that correctly re-plans after the "
        "change is not counted as failing.",
        "",
        "## How fragile is the planner to a single targeted cell?",
        "",
        f"- **{overall['frac_instances_with_a_damaging_cell']:.0%}** of the "
        f"{overall['n_instances_optimal_at_baseline']} maps that were optimal before "
        "the flip have at least one cell whose flip costs optimality.",
        f"- But the damaging cells are rare: a median of "
        f"**{overall['median_frac_cells_damaging']:.2%}** of candidate cells per map "
        f"(mean {overall['mean_frac_cells_damaging']:.2%}).",
        f"- Worst single-flip damage anywhere in the split: Opt ratio "
        f"**{overall['max_worst_opt']:.3f}**; median across maps "
        f"{overall['median_worst_opt']:.3f}.",
        "",
        "## The ten most fragile maps",
        "",
        "| instance | baseline Opt | worst Opt | cell | damaging cells | worst Exp (Opt ≤ 1.05) | cell |",
        "|----------|--------------|-----------|------|----------------|------------------------|------|",
    ]
    for p in worst:
        exp_cell = tuple(p["worst_exp_cell"]) if p["worst_exp_cell"] else "—"
        lines.append(
            f"| {p['instance_id']} | {p['baseline_opt']:.3f} | **{p['worst_opt']:.3f}** | "
            f"{tuple(p['worst_opt_cell'])} | {p['frac_cells_damaging_opt']:.1%} | "
            f"{p['worst_exp_within_cap']:.3f} | {exp_cell} |"
        )
    lines += [
        "",
        "## Selected flips",
        "",
        f"Two alterations are tagged per map in `results/manifest{meta.get('suffix', '')}"
        f".parquet`: the worst "
        f"by Opt ratio (`is_worst_optimality`), and the worst by Exp ratio among "
        f"alterations costing at most {OPT_CAP_FOR_EFFICIENCY:.2f} in Opt "
        "(`is_worst_efficiency`). Step 7 runs the 2×2 swap on these.",
        "",
        f"Manifest columns follow the project schema; the swap columns "
        "(`opt_C0O1`, `opt_C1O0`, `exp_C0O1`, `exp_C1O0`) are added by step 7.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    """Run the sweep (or just time it) and write the manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--instances", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--timing-only", action="store_true")
    parser.add_argument(
        "--family",
        default="mazes",
        choices=list(config.MAP_FAMILIES),
        help="map family; all eight ship with the nested dataset submodule",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="encoder checkpoint; defaults to the shipped pretrained one",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="suffix for the manifest and report, e.g. mazes_seed1",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="manifest path; derived from --tag if omitted"
    )
    args = parser.parse_args()

    split_name = config.canonical_split(args.split)
    config.set_global_seeds(args.seed)
    split = load_split(split_name, npz_path=config.family_npz(args.family))
    instances = split.instances(seed=args.seed)
    n_total = len(instances)
    if args.instances is not None:
        instances = instances[: args.instances]

    suffix = f"_{args.tag}" if args.tag else ""
    if args.out is None:
        args.out = config.RESULTS_DIR / f"manifest{suffix}.parquet"
    planner = load_pretrained_neural_astar(checkpoint_path=args.checkpoint)
    baselines: Dict[int, Baseline] = {}
    rows: List[Dict[str, object]] = []
    started = time.perf_counter()

    for position, instance in enumerate(instances, start=1):
        flips = enumerate_k1_flips(instance)
        requests = [
            AlterationRequest(
                instance=instance, flip_cells=cells, method="exhaustive", objective="none"
            )
            for cells in flips
        ]
        rows.extend(
            evaluate_alterations(
                planner, requests, baselines=baselines, batch_size=args.batch_size,
                seed=args.seed,
            )
        )
        if position % 5 == 0 or position == len(instances):
            elapsed = time.perf_counter() - started
            rate = elapsed / position
            print(
                f"  {position}/{len(instances)} maps | {len(rows):,} alterations | "
                f"{elapsed / 60:.1f} min elapsed | "
                f"{rate * (len(instances) - position) / 60:.1f} min left",
                flush=True,
            )

    seconds = time.perf_counter() - started
    ms_each = 1000 * seconds / max(len(rows), 1)
    print(f"\n{len(rows):,} alterations in {seconds / 60:.1f} min ({ms_each:.1f} ms each)")

    if args.timing_only:
        scale = n_total / max(len(instances), 1)
        print(
            f"extrapolation to all {n_total} maps: {int(len(rows) * scale):,} alterations, "
            f"{seconds * scale / 60:.0f} min"
        )
        return 0

    select_worst_flips(rows)
    summary = summarize_sweep(rows)

    frame = pd.DataFrame(rows)
    for column in ("flip_cells",):
        frame[column] = frame[column].apply(lambda cells: [list(rc) for rc in cells])
    for column in ("start_rc", "goal_rc"):
        frame[column] = frame[column].apply(list)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out, index=False)

    meta = {
        "suffix": suffix,
        "split": split_name,
        "seed": args.seed,
        "minutes": seconds / 60,
        "ms_per_alteration": ms_each,
    }
    (config.RESULTS_DIR / f"e2_k1_sweep{suffix}.json").write_text(
        json.dumps({"meta": meta, **summary}, indent=2, default=float) + "\n"
    )
    (config.RESULTS_DIR / f"e2_k1_sweep{suffix}.md").write_text(
        render_markdown(summary, meta) + "\n"
    )

    overall = summary["overall"]
    print(
        f"\n{overall['frac_instances_with_a_damaging_cell']:.0%} of baseline-optimal maps "
        f"have a damaging cell; median {overall['median_frac_cells_damaging']:.2%} of cells "
        f"are damaging; worst Opt {overall['max_worst_opt']:.3f}"
    )
    print(
        f"wrote {args.out.relative_to(config.REPO_ROOT)} and e2_k1_sweep{suffix}.md"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
