"""PLAN.md step 3: the random-flip pilot (empirical premise of E2).

Flips k ∈ {1, 2, 4, 8} random cells on every test map, with a fixed number of
independent draws per budget, keeping the goal reachable. Each altered map is
evaluated with Neural A\\* (encoder *and* search see the alteration — condition
``C1O1``) and with the exact solver on the **altered** map, which is the
reference a correct planner would match.

Writes `results/pilot_random.parquet` (one row per alteration, schema-compatible
with the step-6 manifest) and `results/pilot_random.md` (the per-budget table
PLAN.md asks for, plus a reading of it).

Usage:
    python scripts/pilot_random.py --instances 3 --timing-only   # extrapolate first
    python scripts/pilot_random.py                              # the full pilot
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
    AlterationRequest,
    Baseline,
    evaluate_alterations,
    sample_random_flips,
    summarize_by_direction,
    summarize_rows,
)
from src.planner import load_pretrained_neural_astar  # noqa: E402

BUDGETS = (1, 2, 4, 8)
EXP_THRESHOLD = 1.5


def build_requests(
    instances, budgets, draws: int, seed: int, free_only: bool
) -> tuple[List[AlterationRequest], List[Dict[str, object]]]:
    """Sample one admissible alteration per (instance, budget, draw).

    The generator is seeded per draw with ``[seed, instance_id, budget, draw]``,
    so a draw is reproducible regardless of how many instances or budgets the
    run covers.

    Args:
        instances: `MazeInstance` objects with fixed starts.
        budgets: Flip budgets to sample.
        draws: Independent draws per (instance, budget).
        seed: Base seed.
        free_only: Restrict flips to free cells (obstacle-adding only).

    Returns:
        tuple: ``(requests, rejections)`` where each rejection records an
        (instance, budget, draw) for which no admissible draw was found.
    """
    requests: List[AlterationRequest] = []
    rejections: List[Dict[str, object]] = []
    for instance in instances:
        for budget in budgets:
            for draw in range(draws):
                rng = np.random.default_rng([seed, instance.instance_id, budget, draw])
                cells = sample_random_flips(instance, budget, rng, free_only=free_only)
                if cells is None:
                    rejections.append(
                        {
                            "instance_id": instance.instance_id,
                            "budget": budget,
                            "draw": draw,
                            "reason": "no admissible draw kept the goal reachable",
                        }
                    )
                    continue
                requests.append(
                    AlterationRequest(
                        instance=instance,
                        flip_cells=cells,
                        method="random",
                        objective="none",
                        draw=draw,
                    )
                )
    return requests, rejections


def decision_section(
    table: List[Dict[str, object]], probe: Dict[str, object] | None
) -> List[str]:
    """Render the budget decision step 3 owes, from the pilot and the probe.

    The pilot measures the average flip; the worst-case probe
    (`scripts/probe_exhaustive_k1.py`, `results/probe_k1.json`) measures the
    worst one. The decision needs both: a low random-damage rate means either a
    robust planner or a weak baseline, and only the probe separates the two.

    Args:
        table: Output of `src.perturb.summarize_rows`.
        probe: Parsed `results/probe_k1.json`, or None if it has not been run.

    Returns:
        list[str]: Markdown lines.
    """
    lines = ["## Decision: which budgets deserve a targeted search", ""]
    k1 = table[0]
    k8 = table[-1]
    lines += [
        f"Random flips are a weak attack. At k=1 only "
        f"{k1['frac_newly_suboptimal']:.0%} of alterations push a "
        f"previously-optimal instance off the optimum, rising to "
        f"{k8['frac_newly_suboptimal']:.0%} at k={k8['budget']}; the efficiency "
        f"failure rate stays at {k1['frac_exp_above_threshold']:.0%} across all "
        f"budgets. Most of the raw `Opt > 1` column is the planner's standing "
        f"error rate, not flip damage.",
        "",
    ]
    if probe is None:
        lines += [
            "The worst-case probe has not been run, so it is not yet possible to "
            "tell a robust planner from a weak baseline. Run "
            "`python scripts/probe_exhaustive_k1.py --instances 3`.",
            "",
        ]
        return lines

    rows = probe["per_instance"]
    worst = [r["worst_opt"] for r in rows]
    damaging = [r["frac_cells_opt_above_1"] for r in rows]
    n_with_damage = sum(1 for w in worst if w > 1.0)
    lines += [
        f"The exhaustive k=1 probe on {probe['n_instances']} instances "
        f"({probe['n_alterations']} alterations, {probe['seconds']:.0f} s) says it "
        "is a weak baseline, not a robust planner:",
        "",
        "| instance | baseline Opt | worst Opt from one flip | cell | share of cells that damage Opt | worst Exp (Opt ≤ 1.05) |",
        "|----------|--------------|-------------------------|------|--------------------------------|------------------------|",
    ]
    for r in rows:
        worst_exp = r["worst_exp_within_opt_cap"]
        lines.append(
            f"| {r['instance_id']} | {r['baseline_opt']:.3f} | **{r['worst_opt']:.3f}** | "
            f"{tuple(r['worst_opt_cell'])} | {r['frac_cells_opt_above_1']:.1%} | "
            f"{worst_exp:.3f} |"
        )
    lines += [
        "",
        f"Every one of the {n_with_damage}/{len(rows)} instances has at least one "
        f"single cell whose flip costs optimality (worst `Opt` "
        f"{min(worst):.3f}–{max(worst):.3f}), but only "
        f"{min(damaging):.1%}–{max(damaging):.1%} of the ~1 000 candidate cells "
        "do — one to a dozen cells per map. A random draw lands on one with "
        f"probability ≈{np.mean(damaging):.1%} per flip, the same order as the "
        f"{k1['frac_newly_suboptimal']:.0%} damage rate the pilot measured at "
        "k=1 — consistent with the two measuring the same thing, though the "
        "probe covers only three maps.",
        "",
        "**Fragility is real but extremely localized.** That is a substantive "
        "result for the project, not just a methodological note: it means the "
        "mediation question of E3 has to be asked about *targeted* flips, "
        "because the average flip carries almost no damage to decompose.",
        "",
        "Consequently:",
        "",
        "* **k=1 exhaustive (step 6): worth it, and necessary.** At "
        f"{probe['ms_per_alteration']:.0f} ms per alteration the full test split "
        f"costs ≈{probe['ms_per_alteration'] * 1022 * 100 / 1000 / 60:.0f} min "
        "single-process. Random search cannot substitute: it would need ~100–1000 "
        "draws per map to hit the same cells.",
        "* **k ∈ {2, 4, 8} greedy (step 9): worth it.** Random damage does grow "
        f"with budget ({k1['frac_newly_suboptimal']:.0%} → "
        f"{k8['frac_newly_suboptimal']:.0%}), so there is signal to climb, and "
        "greedy starts from the k=1 cells this sweep will have found.",
        "* **Random flips stay as the control** at every budget, which is what "
        "PLAN.md § 5 asks for to keep the attack-selection bias visible.",
        "* **The efficiency objective needs watching.** Only "
        f"{sum(1 for r in rows if r['frac_cells_exp_above_threshold'] > 0)}/{len(rows)} "
        "probe instances have any cell pushing `Exp` above 1.5 (one reaches "
        "3.75). If the full sweep confirms that, the efficiency arm of E2 will "
        "have a small sample and the H2 decomposition may only be answerable for "
        "the optimality objective.",
        "",
    ]
    return lines


def render_markdown(
    table: List[Dict[str, object]],
    direction_table: List[Dict[str, object]],
    meta: Dict[str, object],
    rejections: List[Dict[str, object]],
    probe: Dict[str, object] | None = None,
) -> str:
    """Render the step-3 report.

    Args:
        table: Output of `src.perturb.summarize_rows`.
        direction_table: Output of `src.perturb.summarize_by_direction`.
        meta: Run metadata (split, seed, counts, timings).
        rejections: Draws that could not be made admissible.
        probe: Parsed `results/probe_k1.json` if available, for the decision
            section.

    Returns:
        str: Markdown document for `results/pilot_random.md`.
    """
    lines = [
        "# Step 3 — random-flip pilot",
        "",
        f"Split `{meta['split']}` ({meta['n_instances']} instances), "
        f"seed {meta['seed']}, {meta['draws']} random draws per budget, "
        f"{meta['n_rows']} alterations evaluated in {meta['seconds']:.1f} s "
        f"({meta['device']}).",
        "",
        "Flips are drawn uniformly from all cells except the start and the goal "
        "(so a flip may add *or* remove an obstacle), and any draw that "
        "disconnects the goal is redrawn. `Opt` and `Exp` are measured against "
        "the **altered** map's optimum and classical A\\* expansion count — "
        "classical A\\* replanning after a flip is the correct response, not an "
        "error.",
        "",
        "## Fragility by budget",
        "",
        "| k | n | Opt > 1 | newly suboptimal | Exp > 1.5 | newly Exp > 1.5 | mean Opt | mean Exp |",
        "|---|---|---------|------------------|-----------|-----------------|----------|----------|",
    ]
    for row in table:
        lines.append(
            f"| {row['budget']} | {row['n']} | {row['frac_opt_deviating']:.0%} | "
            f"{row['frac_newly_suboptimal']:.0%} | {row['frac_exp_above_threshold']:.0%} | "
            f"{row['frac_newly_exp_above_threshold']:.0%} | {row['mean_opt_C1O1']:.4f} | "
            f"{row['mean_exp_C1O1']:.4f} |"
        )
    lines += [
        "",
        "## What the flip did to the map",
        "",
        "| k | optimum changed | optimum got shorter | mean Opt when newly broken | worst |",
        "|---|-----------------|---------------------|----------------------------|-------|",
    ]
    for row in table:
        lines.append(
            f"| {row['budget']} | {row['frac_optimum_changed']:.1%} | "
            f"{row['frac_optimum_shorter']:.1%} | "
            f"{row['mean_opt_when_newly_broken']:.4f} | "
            f"{row['max_opt_when_newly_broken']:.4f} |"
        )
    lines += [
        "",
        f"At k=1 the optimal cost changes in only "
        f"{table[0]['frac_optimum_changed']:.0%} of draws, so every k=1 failure "
        "is the planner deviating on a map whose optimum it already matched — no "
        "\"the map genuinely changed\" confound. Damage, when it happens, is not "
        f"marginal: the newly-broken alterations average "
        f"{table[0]['mean_opt_when_newly_broken']:.3f} and reach "
        f"{max(r['max_opt_when_newly_broken'] for r in table):.3f} across budgets "
        "(a third longer than the optimum).",
        "",
        "## Pillars vs doors (k=1)",
        "",
        "| flip direction | n | share of draws | newly suboptimal | mean Opt | mean ΔCost L1 |",
        "|----------------|---|----------------|------------------|----------|---------------|",
    ]
    for row in direction_table:
        lines.append(
            f"| {row['direction']} | {row['n']} | {row['share_of_draws']:.1%} | "
            f"{row['frac_newly_suboptimal']:.1%} | {row['mean_opt_C1O1']:.4f} | "
            f"{row['mean_dcost_l1']:.3f} |"
        )
    lines += [
        "",
        "The maps are ~11% obstacles, with one-cell-thick walls separating open "
        "areas roughly ten cells wide (`docs/verified.md` § 11). So the two flip "
        "directions are different edits: making a free cell an obstacle drops an "
        "isolated **pillar** into open space, which an 8-connected search walks "
        "around at no cost, while making an obstacle free opens a **door** "
        "between two areas. Doors are rarer (they are only 11% of cells) but "
        "several times more damaging per flip — which is the main reason uniform "
        "random flipping is such a weak attack.",
        "",
        "* **Opt > 1** — share of alterations where Neural A\\*'s path is longer "
        "than the altered map's optimum (PLAN.md's headline number).",
        "* **newly suboptimal** — the same share, restricted to alterations whose "
        "unaltered baseline was already optimal. This is the damage *caused* by "
        "the flip, separated from the planner's standing error rate "
        f"(baseline mean Opt {table[0]['mean_opt_C0O0']:.4f}, "
        f"mean Exp {table[0]['mean_exp_C0O0']:.4f}).",
        "* **Exp > 1.5** — share expanding more than 1.5x classical A\\* on the "
        "altered map.",
        "",
        "## ΔCost in the encoder output",
        "",
        "| k | mean ΔCost L1 | mean share of ΔCost mass under the flipped cells |",
        "|---|---------------|---------------------------------------------------|",
    ]
    for row in table:
        lines.append(
            f"| {row['budget']} | {row['mean_dcost_l1']:.3f} | "
            f"{row['mean_dcost_frac_under_flips']:.1%} |"
        )
    lines += [
        "",
        f"A flipped cell changes the cost map over a neighbourhood, not only at "
        f"the flipped cell: with k=1 only "
        f"{table[0]['mean_dcost_frac_under_flips']:.1%} of the total |Δcost| mass "
        f"sits on the flipped cell itself. The spread is bounded exactly: the "
        f"encoder is five 3x3 convolutions, so its receptive field has radius 5, "
        f"and |Δcost| from a k=1 flip is *exactly zero* outside the 11x11 window "
        f"around the flipped cell (verified in docs/verified.md section 10). Any "
        f"behaviour change further away travelled through the search, not "
        f"through the encoder's view of that region.",
        "",
    ]
    lines += decision_section(table, probe)
    if rejections:
        lines += [
            "## Rejected draws",
            "",
            f"{len(rejections)} (instance, budget, draw) combinations had no "
            "admissible draw within the attempt cap and were skipped:",
            "",
            "```",
            *[json.dumps(r) for r in rejections[:20]],
            "```",
            "",
        ]
    return "\n".join(lines)


def main() -> int:
    """Run the pilot (or just the timing probe) and write the artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--draws", type=int, default=3, help="random draws per budget")
    parser.add_argument(
        "--instances", type=int, default=None, help="limit to the first N instances"
    )
    parser.add_argument(
        "--free-only",
        action="store_true",
        help="only flip free cells (obstacle-adding alterations)",
    )
    parser.add_argument(
        "--timing-only",
        action="store_true",
        help="print timings and an extrapolation to the full split, write nothing",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    split_name = config.canonical_split(args.split)
    config.set_global_seeds(args.seed)
    split = load_split(split_name)
    instances = split.instances(seed=args.seed)
    n_total = len(instances)
    if args.instances is not None:
        instances = instances[: args.instances]

    planner = load_pretrained_neural_astar()

    started = time.perf_counter()
    requests, rejections = build_requests(
        instances, BUDGETS, args.draws, args.seed, args.free_only
    )
    sampling_seconds = time.perf_counter() - started

    baselines: Dict[int, Baseline] = {}
    started = time.perf_counter()
    rows = evaluate_alterations(
        planner, requests, baselines=baselines, batch_size=args.batch_size, seed=args.seed
    )
    eval_seconds = time.perf_counter() - started
    total_seconds = sampling_seconds + eval_seconds

    print(
        f"\n{len(instances)}/{n_total} instances, budgets {list(BUDGETS)}, "
        f"{args.draws} draws -> {len(rows)} alterations"
    )
    print(
        f"  sampling {sampling_seconds:.2f} s | evaluation {eval_seconds:.2f} s "
        f"| total {total_seconds:.2f} s "
        f"({1000 * total_seconds / max(len(rows), 1):.1f} ms per alteration)"
    )
    if rejections:
        print(f"  {len(rejections)} draws rejected (goal unreachable)")

    if args.timing_only:
        scale = n_total / max(len(instances), 1)
        print(
            f"\nextrapolation to all {n_total} instances: "
            f"{int(len(rows) * scale)} alterations, "
            f"{total_seconds * scale:.0f} s "
            f"({total_seconds * scale / 60:.1f} min)"
        )
        print(
            "for reference, PLAN.md step 6 (exhaustive k=1, ~1022 cells x "
            f"{n_total} maps) would be "
            f"{1022 * n_total * total_seconds / max(len(rows), 1) / 60:.0f} min "
            "at this per-alteration cost"
        )
        return 0

    frame = pd.DataFrame(rows)
    frame["flip_cells"] = frame["flip_cells"].apply(lambda cells: [list(rc) for rc in cells])
    frame["start_rc"] = frame["start_rc"].apply(list)
    frame["goal_rc"] = frame["goal_rc"].apply(list)
    parquet_path = config.RESULTS_DIR / "pilot_random.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(parquet_path, index=False)

    invalid = frame.loc[~frame["path_valid_C1O1"]]
    if len(invalid):
        print(f"\nWARNING: {len(invalid)} alterations produced an invalid path")
        print(invalid[["instance_id", "budget", "path_invalid_reason"]].head().to_string())

    table = summarize_rows(rows, exp_threshold=EXP_THRESHOLD)
    meta = {
        "split": split_name,
        "seed": args.seed,
        "draws": args.draws,
        "n_instances": len(instances),
        "n_rows": len(rows),
        "seconds": total_seconds,
        "device": str(config.get_device()),
    }
    probe_path = config.RESULTS_DIR / "probe_k1.json"
    probe = json.loads(probe_path.read_text()) if probe_path.exists() else None
    md_path = config.RESULTS_DIR / "pilot_random.md"
    direction_table = summarize_by_direction(rows, budget=1)
    md_path.write_text(
        render_markdown(table, direction_table, meta, rejections, probe) + "\n"
    )

    print("\n| k | n | Opt>1 | newly subopt | Exp>1.5 | newly Exp>1.5 | mean Opt | mean Exp |")
    print("|---|---|-------|--------------|---------|---------------|----------|----------|")
    for row in table:
        print(
            f"| {row['budget']} | {row['n']} | {row['frac_opt_deviating']:.0%} | "
            f"{row['frac_newly_suboptimal']:.0%} | {row['frac_exp_above_threshold']:.0%} | "
            f"{row['frac_newly_exp_above_threshold']:.0%} | {row['mean_opt_C1O1']:.4f} | "
            f"{row['mean_exp_C1O1']:.4f} |"
        )
    print(
        f"\nwrote {parquet_path.relative_to(config.REPO_ROOT)} and "
        f"{md_path.relative_to(config.REPO_ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
