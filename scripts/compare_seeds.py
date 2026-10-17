"""PLAN.md step 10: repeat the key measurements across training seeds.

A single trained model cannot tell you which of its properties are facts about
Neural A\\* and which are accidents of one optimisation run. This script takes
several checkpoints and repeats, for each: the reproduction metrics, the
surrogate fits of step 5 with the sufficiency test, and — if a per-seed
exhaustive sweep exists — the mediation decomposition of step 7.

It reports the **dispersion**, because that is the point: a result that moves
by more between seeds than between conditions is not a result.

Usage:
    python scripts/compare_seeds.py --tags "" seed0 seed1 seed2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.data import load_split  # noqa: E402
from src.exact_solver import solve_astar, solve_dijkstra  # noqa: E402
from src.metrics import (  # noqa: E402
    exp_ratio,
    harmonic_mean,
    n_expansions,
    opt_ratio,
    path_cost,
)
from src.planner import load_pretrained_neural_astar  # noqa: E402
from src.surrogate import (  # noqa: E402
    build_design_matrix,
    fit_and_score,
    load_archive,
    sufficiency_cost_maps,
)

#: Surrogate fits repeated per seed; the groups that carry the step-5 argument.
GROUPS = (("geometric_h1", "gbm"), ("path", "gbm"), ("all", "gbm"))


def planning_metrics(planner, instances, cost_maps: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Opt / Exp / Hmean for a planner, or for a supplied cost map.

    Args:
        planner: A `NeuralAstar`; its search module is always used.
        instances: The instances to evaluate.
        cost_maps: Optional ``(N, H, W)`` cost maps to search with instead of
            the encoder's own output.

    Returns:
        dict[str, float]: ``p_opt``, ``mean_opt``, ``mean_exp`` and ``h_mean``.
    """
    from src.planner import run_astar_module

    tensors = [inst.to_tensors() for inst in instances]
    maps = torch.cat([t[0] for t in tensors])
    starts = torch.cat([t[1] for t in tensors])
    goals = torch.cat([t[2] for t in tensors])
    with torch.no_grad():
        if cost_maps is None:
            out = planner(maps, starts, goals)
        else:
            costs = torch.from_numpy(cost_maps.astype(np.float32)).unsqueeze(1)
            out = run_astar_module(planner, costs, starts, goals, maps)

    optimum = [solve_dijkstra(i.map_design, i.start_rc, i.goal_rc).cost for i in instances]
    classical = [
        float(solve_astar(i.map_design, i.start_rc, i.goal_rc).expansions) for i in instances
    ]
    ratios_opt = opt_ratio(path_cost(out.paths), np.array(optimum))
    ratios_exp = exp_ratio(n_expansions(out.histories), np.array(classical))
    p_opt = float(np.mean(ratios_opt <= 1.0))
    p_exp = float(np.nanmean(np.maximum(1.0 - ratios_exp, 0.0)))
    return {
        "p_opt": p_opt,
        "mean_opt": float(np.nanmean(ratios_opt)),
        "mean_exp": float(np.nanmean(ratios_exp)),
        "h_mean": harmonic_mean(p_opt, p_exp),
    }


def checkpoint_epoch(checkpoint: Optional[Path]) -> Optional[int]:
    """Epoch recorded in a checkpoint's filename, if it says.

    Worth reporting: the shipped checkpoint stopped at epoch 33 while models
    trained here run to 50, so a difference between it and the retrained set
    is not purely seed variance.

    Args:
        checkpoint: Checkpoint file or directory; None means the shipped one.

    Returns:
        int | None: The epoch number, or None if it cannot be read.
    """
    import re

    base = Path(checkpoint) if checkpoint is not None else config.PRETRAINED_CKPT_DIR
    files = sorted(base.glob("**/*.ckpt")) if base.is_dir() else [base]
    if not files:
        return None
    match = re.search(r"epoch=(\d+)", files[-1].name)
    return int(match.group(1)) if match else None


def evaluate_tag(tag: str, checkpoint: Optional[Path], seed: int) -> Dict[str, object]:
    """Run every per-seed measurement for one model.

    Args:
        tag: Archive tag for this model (empty for the shipped checkpoint).
        checkpoint: Checkpoint directory or file; None uses the shipped one.
        seed: Start-selection seed, held fixed across models so the instances
            are identical.

    Returns:
        dict: Planning metrics, surrogate R² per group, and the sufficiency
        numbers for the full surrogate.

    Raises:
        FileNotFoundError: If the feature archive for this tag is missing.
    """
    planner = load_pretrained_neural_astar(checkpoint_path=checkpoint)
    split = load_split("test")
    instances = split.instances(seed=seed)

    archive = load_archive("test", tag=tag)
    train = build_design_matrix("train", tag=tag)
    test = build_design_matrix("test", archive=archive)

    fits = {}
    for group, kind in GROUPS:
        result = fit_and_score(train, test, group, kind)
        fits[f"{group}_{kind}"] = result.r2
        if group == "all":
            full = result
    out = {
        "tag": tag or "pretrained",
        "checkpoint_epoch": checkpoint_epoch(checkpoint),
        "planning": planning_metrics(planner, instances),
        "surrogate_r2": fits,
        "sufficiency_all": planning_metrics(
            planner, instances, sufficiency_cost_maps(full, archive)
        ),
    }
    out.update(mediation_for(tag))
    out.update(fragility_for(tag))
    return out


def fragility_for(tag: str) -> Dict[str, object]:
    """Read this model's exhaustive-sweep summary, if its sweep has been run.

    Separated from the mediation numbers because the two behave very
    differently across models: how fragile a planner is turns out to be stable,
    while how the damage decomposes does not.

    Args:
        tag: Model tag (empty for the shipped checkpoint).

    Returns:
        dict: Share of baseline-optimal maps with a damaging cell, the median
        and mean share of damaging cells, and the worst single-flip damage.
        Empty when the model's sweep output is absent.
    """
    suffix = f"_{tag}" if tag else ""
    path = config.RESULTS_DIR / f"e2_k1_sweep{suffix}.json"
    if not path.exists():
        return {}
    overall = json.loads(path.read_text())["overall"]
    return {
        "fragility": {
            "maps_with_a_damaging_cell": overall["frac_instances_with_a_damaging_cell"],
            "median_damaging_cells": overall["median_frac_cells_damaging"],
            "mean_damaging_cells": overall["mean_frac_cells_damaging"],
            "worst_opt": overall["max_worst_opt"],
            "median_worst_opt": overall["median_worst_opt"],
        }
    }


def mediation_for(tag: str) -> Dict[str, object]:
    """Read this model's step-7 decomposition, if its swap has been run.

    Each model needs its own sweep and swap, because each has its own worst
    flips; this only collects what `scripts/seed_pipeline.sh` produced.

    Args:
        tag: Model tag (empty for the shipped checkpoint).

    Returns:
        dict: Mediation means per objective, restricted to the alterations
        that did not change the map's optimum — the clean failure mode. Empty
        when the model's swap output is absent.
    """
    suffix = f"_{tag}" if tag else ""
    path = config.RESULTS_DIR / f"e3_k1{suffix}.json"
    if not path.exists():
        return {}
    report = json.loads(path.read_text())
    split = report.get("summaries_by_failure_mode", {})
    out: Dict[str, object] = {}
    for objective in ("optimality", "efficiency"):
        part = split.get(objective, {}).get("optimum unchanged")
        if not part or not part.get("n_with_damage"):
            continue
        out[f"mediation_{objective}"] = {
            "n": part["n_with_damage"],
            "encoder": part["encoder_mean"],
            "search": part["search_mean"],
        }
    return out


def vulnerability_overlap(tags: List[str]) -> List[Dict[str, object]]:
    """Do different models break on the same cells?

    Each model's exhaustive sweep labels every one of the ~102 000 single-cell
    alterations as damaging or not. Comparing those labels across models
    separates a property of the *map* — this cell is dangerous for any planner
    — from a property of the *model*.

    Args:
        tags: Model tags whose manifests to compare pairwise.

    Returns:
        list[dict]: Per pair, how many cells damage each model, how many damage
        both, the enrichment over what independence would predict, and how
        often the per-map worst flip is the same cell.
    """
    frames = {}
    for tag in tags:
        suffix = f"_{tag}" if tag else ""
        path = config.RESULTS_DIR / f"manifest{suffix}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        frame = frame[(frame["method"] == "exhaustive") & (frame["budget"] == 1)].copy()
        frame["cell"] = frame["flip_cells"].apply(
            lambda cells: (int(cells[0][0]), int(cells[0][1]))
        )
        frame["damaging"] = frame["opt_C1O1"] > frame["opt_C0O0"]
        frames[tag] = frame

    out = []
    names = list(frames)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            a, b = frames[first], frames[second]
            merged = a[["instance_id", "cell", "damaging", "is_worst_optimality"]].merge(
                b[["instance_id", "cell", "damaging", "is_worst_optimality"]],
                on=["instance_id", "cell"],
                suffixes=("_a", "_b"),
            )
            both = int((merged["damaging_a"] & merged["damaging_b"]).sum())
            n_a, n_b = int(merged["damaging_a"].sum()), int(merged["damaging_b"].sum())
            expected = n_a * n_b / max(len(merged), 1)
            worst_a = merged[merged["is_worst_optimality_a"].fillna(False).astype(bool)]
            same_worst = int(
                worst_a["is_worst_optimality_b"].fillna(False).astype(bool).sum()
            )
            out.append(
                {
                    "pair": f"{first or 'pretrained'} vs {second or 'pretrained'}",
                    "damaging_first": n_a,
                    "damaging_second": n_b,
                    "damaging_both": both,
                    "share_of_first_shared": both / n_a if n_a else float("nan"),
                    "enrichment_over_chance": both / expected if expected else float("nan"),
                    "same_worst_flip_maps": same_worst,
                    "n_maps": int(len(worst_a)),
                }
            )
    return out


def cost_map_agreement(tags: List[str]) -> List[Dict[str, object]]:
    """How much do two models' cost maps agree, cell by cell?

    The cost map is the object this whole project interprets. If two models
    trained the same way produce very different ones, then statements about
    "the" cost map are statements about one optimisation run.

    Correlations are computed over free, reachable cells only — the ones the
    search can actually visit — and also within each instance, since the
    pooled figure could be carried by between-map differences.

    Args:
        tags: Model tags to compare pairwise.

    Returns:
        list[dict]: One entry per pair with the pooled and mean within-map
        correlation and the mean absolute difference.
    """
    archives = {}
    for tag in tags:
        try:
            archives[tag] = load_archive("test", tag=tag)
        except FileNotFoundError:
            continue
    out = []
    names = list(archives)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            a, b = archives[first], archives[second]
            mask = (a["is_free"] > 0) & (a["reachable"] > 0)
            x, y = a["cost_map"][mask].astype(np.float64), b["cost_map"][mask].astype(np.float64)
            per_instance = []
            for index in range(a["cost_map"].shape[0]):
                cells = mask[index]
                u = a["cost_map"][index][cells].astype(np.float64)
                v = b["cost_map"][index][cells].astype(np.float64)
                if u.std() > 0 and v.std() > 0:
                    per_instance.append(float(np.corrcoef(u, v)[0, 1]))
            out.append(
                {
                    "pair": f"{first or 'pretrained'} vs {second or 'pretrained'}",
                    "pooled_r": float(np.corrcoef(x, y)[0, 1]),
                    "within_map_r": float(np.mean(per_instance)) if per_instance else float("nan"),
                    "mean_abs_difference": float(np.abs(x - y).mean()),
                }
            )
    return out


def mediation_on_shared_flips(
    tag: str, checkpoint: Optional[Path], seed: int, manifest: Path
) -> Dict[str, object]:
    """Decompose the *same* alterations with every model.

    Repeating step 7 per model uses each model's own worst flips, which is the
    faithful repetition but mixes two sources of variance: the models differ,
    *and* the alterations differ. Running every model on one fixed set of
    flips isolates the first.

    The shared set is the alterations tagged in the reference manifest, which
    were chosen as worst for the shipped checkpoint — so this measurement is
    "how do other models respond to the flips that break this one", not "how
    fragile is each model".

    Args:
        tag: Model tag (empty for the shipped checkpoint).
        checkpoint: Its checkpoint, or None for the shipped one.
        seed: Start-selection seed.
        manifest: Manifest holding the tagged flips.

    Returns:
        dict: Mediation means per objective over the shared alterations,
        restricted to those where the optimum did not change.
    """
    from src.swap import run_swap, summarize_swaps

    frame = pd.read_parquet(manifest)
    planner = load_pretrained_neural_astar(checkpoint_path=checkpoint)
    split = load_split("test")
    out: Dict[str, object] = {}
    for objective, column, metric in (
        ("optimality", "is_worst_optimality", "opt"),
        ("efficiency", "is_worst_efficiency", "exp"),
    ):
        selected = frame[frame[column].fillna(False).astype(bool)]
        results = []
        for _, record in selected.iterrows():
            instance = split.instance(int(record["instance_id"]), seed=seed)
            cells = [tuple(int(v) for v in cell) for cell in record["flip_cells"]]
            results.append(run_swap(planner, instance, cells))
        clean = [r for r in results if not r.optimum_changed]
        summary = summarize_swaps(clean, metric)
        if summary["n_with_damage"]:
            out[f"shared_{objective}"] = {
                "n": summary["n_with_damage"],
                "encoder": summary["encoder_mean"],
                "search": summary["search_mean"],
            }
    return out


def main() -> int:
    """Compare every requested model and report the dispersion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tags", nargs="+", default=[""], help="archive tags to compare")
    parser.add_argument(
        "--checkpoints", nargs="*", default=[], help="checkpoint per tag, in the same order"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--shared-flips",
        type=Path,
        default=None,
        help="manifest whose tagged flips every model should also be measured on",
    )
    args = parser.parse_args()

    # An empty string means "the shipped checkpoint". Converting it to Path("")
    # would yield Path("."), whose recursive glob would pick up a *trained*
    # checkpoint from results/ and silently evaluate the wrong model.
    checkpoints: List[Optional[Path]] = [Path(c) if c else None for c in args.checkpoints]
    while len(checkpoints) < len(args.tags):
        checkpoints.append(None)

    config.set_global_seeds(args.seed)
    rows = []
    for tag, checkpoint in zip(args.tags, checkpoints):
        print(f"evaluating {tag or 'pretrained'}...", flush=True)
        row = evaluate_tag(tag, checkpoint, args.seed)
        if args.shared_flips is not None and args.shared_flips.exists():
            row.update(
                mediation_on_shared_flips(tag, checkpoint, args.seed, args.shared_flips)
            )
        rows.append(row)

    frame = pd.json_normalize(rows)
    numeric = frame.select_dtypes("number")
    dispersion = pd.DataFrame(
        {"mean": numeric.mean(), "std": numeric.std(), "min": numeric.min(), "max": numeric.max()}
    )
    agreement = cost_map_agreement(list(args.tags))
    overlap = vulnerability_overlap(list(args.tags))
    report = {
        "seed": args.seed,
        "models": rows,
        "dispersion": dispersion.to_dict("index"),
        "cost_map_agreement": agreement,
        "vulnerability_overlap": overlap,
    }
    (config.RESULTS_DIR / "seeds.json").write_text(
        json.dumps(report, indent=2, default=float) + "\n"
    )
    (config.RESULTS_DIR / "seeds.md").write_text(
        render_markdown(rows, dispersion, agreement, overlap) + "\n"
    )

    print()
    print(dispersion.round(4).to_string())
    print("\nwrote seeds.md and seeds.json")
    return 0


def render_markdown(
    rows: List[Dict[str, object]],
    dispersion: pd.DataFrame,
    agreement: Optional[List[Dict[str, object]]] = None,
    overlap: Optional[List[Dict[str, object]]] = None,
) -> str:
    """Render the seed comparison.

    Args:
        rows: Per-model results.
        dispersion: Mean / std / min / max per numeric field.
        agreement: Pairwise cost-map agreement.
        overlap: Pairwise agreement on which cells are dangerous.

    Returns:
        str: Markdown document.
    """
    lines = [
        "# Step 10 — dispersion across training seeds",
        "",
        f"{len(rows)} model(s) compared on the same instances and the same fixed starts. "
        "A difference between conditions is only a result if it is larger than the "
        "difference between seeds.",
        "",
        "| model | epoch | exactly optimal | mean Exp | Hmean | R² geometry | R² path | R² all | sufficiency Opt |",
        "|-------|-------|-----------------|----------|-------|-------------|---------|--------|-----------------|",
    ]
    for row in rows:
        planning = row["planning"]
        fits = row["surrogate_r2"]
        sufficiency = row["sufficiency_all"]
        epoch = row.get("checkpoint_epoch")
        lines.append(
            f"| `{row['tag']}` | {epoch if epoch is not None else '—'} | "
            f"{planning['p_opt']:.3f} | {planning['mean_exp']:.3f} | "
            f"{planning['h_mean']:.3f} | {fits['geometric_h1_gbm']:.3f} | "
            f"{fits['path_gbm']:.3f} | {fits['all_gbm']:.3f} | {sufficiency['p_opt']:.3f} |"
        )
    if any("fragility" in row for row in rows):
        lines += [
            "",
            "## Is the planner equally fragile, model to model?",
            "",
            "From each model's own exhaustive sweep of all ~102 000 single-cell "
            "alterations.",
            "",
            "| model | maps with a damaging cell | median damaging cells | worst Opt |",
            "|-------|---------------------------|-----------------------|-----------|",
        ]
        for row in rows:
            part = row.get("fragility")
            if not part:
                continue
            lines.append(
                f"| `{row['tag']}` | {part['maps_with_a_damaging_cell']:.1%} | "
                f"{part['median_damaging_cells']:.2%} | {part['worst_opt']:.3f} |"
            )
        lines.append("")
    if any("shared_optimality" in row for row in rows):
        lines += [
            "",
            "## The same alterations, decomposed by every model",
            "",
            "Repeating step 7 per model uses each model's own worst flips, which mixes "
            "two sources of variance: the models differ and the alterations differ. "
            "Here every model is run on one fixed set — the flips chosen as worst for "
            "the shipped checkpoint — which isolates the model difference.",
            "",
            "| model | n damaged | path: encoder | path: search | n damaged | effort: encoder | effort: search |",
            "|-------|-----------|---------------|--------------|-----------|-----------------|----------------|",
        ]
        for row in rows:
            path_part = row.get("shared_optimality")
            effort_part = row.get("shared_efficiency")
            if not path_part:
                continue
            lines.append(
                f"| `{row['tag']}` | {path_part['n']} | {path_part['encoder']:+.3f} | "
                f"{path_part['search']:+.3f} | "
                + (
                    f"{effort_part['n']} | {effort_part['encoder']:+.3f} | "
                    f"{effort_part['search']:+.3f} |"
                    if effort_part
                    else "— | — | — |"
                )
            )
        lines += [
            "",
            "**Read the `n` columns before the fractions.** The alterations are the "
            "same, but how many of them actually damage each model is not: a flip "
            "chosen as worst for one model often leaves another untouched. So the "
            "fractions are still averages over different subsets, and only the "
            "comparison of *which route dominates* is meaningful — not the exact "
            "values.",
            "",
        ]
    if any("mediation_optimality" in row for row in rows):
        lines += [
            "",
            "## Mediation per model (clean failure mode only)",
            "",
            "Each model has its own worst flips, so each needs its own sweep and swap. "
            "Restricted to alterations that did not change the map's optimum.",
            "",
            "| model | path: encoder | path: search | effort: encoder | effort: search |",
            "|-------|---------------|--------------|-----------------|----------------|",
        ]
        for row in rows:
            path_part = row.get("mediation_optimality")
            effort_part = row.get("mediation_efficiency")
            if not path_part:
                continue
            lines.append(
                f"| `{row['tag']}` | {path_part['encoder']:+.3f} | {path_part['search']:+.3f} | "
                + (
                    f"{effort_part['encoder']:+.3f} | {effort_part['search']:+.3f} |"
                    if effort_part
                    else "— | — |"
                )
            )
    if agreement:
        lines += [
            "",
            "## Do the models agree on the cost map itself?",
            "",
            "The cost map is the object this project interprets. Correlations over "
            "free, reachable cells; the within-map column removes any contribution "
            "from models simply disagreeing about which maps are hard.",
            "",
            "| pair | pooled r | within-map r | mean \\|difference\\| |",
            "|------|----------|--------------|---------------------|",
        ]
        for entry in agreement:
            lines.append(
                f"| {entry['pair']} | {entry['pooled_r']:.3f} | "
                f"{entry['within_map_r']:.3f} | {entry['mean_abs_difference']:.3f} |"
            )
    if overlap:
        lines += [
            "",
            "## Do the models break on the same cells?",
            "",
            "Every model's sweep labels all ~102 000 single-cell alterations as "
            "damaging or not. Comparing the labels separates a property of the map — "
            "this cell is dangerous for any planner — from a property of the model.",
            "",
            "| pair | damaging for each | damaging for both | share | vs chance | same worst flip |",
            "|------|-------------------|-------------------|-------|-----------|-----------------|",
        ]
        for entry in overlap:
            lines.append(
                f"| {entry['pair']} | {entry['damaging_first']:,} / "
                f"{entry['damaging_second']:,} | {entry['damaging_both']:,} | "
                f"{entry['share_of_first_shared']:.0%} | "
                f"**{entry['enrichment_over_chance']:.0f}x** | "
                f"{entry['same_worst_flip_maps']}/{entry['n_maps']} maps |"
            )
        lines.append("")
    lines += ["", "## Dispersion", "", "| quantity | mean | std | min | max |", "|---|---|---|---|---|"]
    for name, row in dispersion.iterrows():
        lines.append(
            f"| `{name}` | {row['mean']:.4f} | {row['std']:.4f} | {row['min']:.4f} | "
            f"{row['max']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
