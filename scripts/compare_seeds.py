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
        "planning": planning_metrics(planner, instances),
        "surrogate_r2": fits,
        "sufficiency_all": planning_metrics(
            planner, instances, sufficiency_cost_maps(full, archive)
        ),
    }
    out.update(mediation_for(tag))
    return out


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


def main() -> int:
    """Compare every requested model and report the dispersion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tags", nargs="+", default=[""], help="archive tags to compare")
    parser.add_argument(
        "--checkpoints", nargs="*", default=[], help="checkpoint per tag, in the same order"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    checkpoints: List[Optional[Path]] = [Path(c) for c in args.checkpoints]
    while len(checkpoints) < len(args.tags):
        checkpoints.append(None)

    config.set_global_seeds(args.seed)
    rows = []
    for tag, checkpoint in zip(args.tags, checkpoints):
        print(f"evaluating {tag or 'pretrained'}...", flush=True)
        rows.append(evaluate_tag(tag, checkpoint, args.seed))

    frame = pd.json_normalize(rows)
    numeric = frame.select_dtypes("number")
    dispersion = pd.DataFrame(
        {"mean": numeric.mean(), "std": numeric.std(), "min": numeric.min(), "max": numeric.max()}
    )
    agreement = cost_map_agreement(list(args.tags))
    report = {
        "seed": args.seed,
        "models": rows,
        "dispersion": dispersion.to_dict("index"),
        "cost_map_agreement": agreement,
    }
    (config.RESULTS_DIR / "seeds.json").write_text(
        json.dumps(report, indent=2, default=float) + "\n"
    )
    (config.RESULTS_DIR / "seeds.md").write_text(
        render_markdown(rows, dispersion, agreement) + "\n"
    )

    print()
    print(dispersion.round(4).to_string())
    print("\nwrote seeds.md and seeds.json")
    return 0


def render_markdown(
    rows: List[Dict[str, object]],
    dispersion: pd.DataFrame,
    agreement: Optional[List[Dict[str, object]]] = None,
) -> str:
    """Render the seed comparison.

    Args:
        rows: Per-model results.
        dispersion: Mean / std / min / max per numeric field.
        agreement: Pairwise cost-map agreement.

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
        "| model | exactly optimal | mean Exp | Hmean | R² geometry | R² path | R² all | sufficiency Opt |",
        "|-------|-----------------|----------|-------|-------------|---------|--------|-----------------|",
    ]
    for row in rows:
        planning = row["planning"]
        fits = row["surrogate_r2"]
        sufficiency = row["sufficiency_all"]
        lines.append(
            f"| `{row['tag']}` | {planning['p_opt']:.3f} | {planning['mean_exp']:.3f} | "
            f"{planning['h_mean']:.3f} | {fits['geometric_h1_gbm']:.3f} | "
            f"{fits['path_gbm']:.3f} | {fits['all_gbm']:.3f} | {sufficiency['p_opt']:.3f} |"
        )
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
