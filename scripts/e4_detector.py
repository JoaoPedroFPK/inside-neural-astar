"""PLAN.md step 12 (E4): can an altered map be detected from the planner's internals?

Every earlier step measured *damage*: how much worse the planner gets when a
few cells are flipped. This step asks the complementary question. The planner
never announces that it has been given a tampered map — it returns a path, and
the path happens to be worse. Is the tampering visible from inside?

Two detectors are compared, both described in `src/detect.py`:

* **surrogate residual** — the cost map minus what the step-5 surrogate
  predicts from map geometry alone, summarised by seven statistics, fed to a
  logistic regression.
* **Mahalanobis on encoder activations** — the standard out-of-distribution
  score, fitted on clean maps only. The comparator: if it wins, the
  interpretability machinery buys nothing here.

**The task is deliberately the easy version.** For each instance the detector
is shown the clean map and the altered one, and judged on separating them.
A real monitor would see one map with no clean twin. Numbers here are an upper
bound on what a deployed detector could do.

**Honest split.** Instances are divided in half by index. Both detectors are
fitted on the first half and scored on the second; the Mahalanobis mean and
covariance use only the *clean* maps of the first half.

Usage:
    python scripts/e4_detector.py --conditions k1_opt --instances 20
    python scripts/e4_detector.py                       # every condition
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.data import load_split  # noqa: E402
from src.detect import (  # noqa: E402
    MAP_ONLY_FEATURES,
    RESIDUAL_FEATURES,
    bootstrap_auc_gap,
    build_examples,
    evaluate_detector,
    evaluate_mahalanobis,
    fit_detector,
    paired_alteration,
    unpaired_mask,
)
from src.features import compute_instance_features  # noqa: E402
from src.planner import load_pretrained_neural_astar  # noqa: E402
from src.surrogate import FEATURE_GROUPS, build_design_matrix, fit_and_score  # noqa: E402

#: Alteration sets to detect. Each entry selects manifest rows; `budget` is the
#: number of flipped cells and `flag` picks the worst alteration per instance
#: for exhaustive sweeps, which record every candidate rather than only the
#: chosen one.
CONDITIONS: Dict[str, Dict[str, object]] = {
    "k1_opt": {"method": "exhaustive", "budget": 1, "flag": "is_worst_optimality"},
    "k1_eff": {"method": "exhaustive", "budget": 1, "flag": "is_worst_efficiency"},
    "greedy_opt_b1": {"method": "greedy", "objective": "optimality", "budget": 1},
    "greedy_opt_b4": {"method": "greedy", "objective": "optimality", "budget": 4},
    "greedy_opt_b8": {"method": "greedy", "objective": "optimality", "budget": 8},
    "greedy_eff_b1": {"method": "greedy", "objective": "efficiency", "budget": 1},
    "greedy_eff_b4": {"method": "greedy", "objective": "efficiency", "budget": 4},
    "greedy_eff_b8": {"method": "greedy", "objective": "efficiency", "budget": 8},
    #: Built rather than looked up. Both take the worst single flip and add a
    #: second one elsewhere: `balanced_2` frees an obstacle, leaving the
    #: obstacle count unchanged, and `blocked_2` blocks another free cell. The
    #: pair isolates the effect of hiding the obstacle count from the effect of
    #: simply flipping one more cell.
    "balanced_2": {"method": "exhaustive", "budget": 1, "flag": "is_worst_optimality",
                   "polarity": "free"},
    "blocked_2": {"method": "exhaustive", "budget": 1, "flag": "is_worst_optimality",
                  "polarity": "block"},
}


def select_alterations(manifest: pd.DataFrame, condition: str) -> Dict[int, List[Tuple[int, int]]]:
    """Pick one alteration per instance for one condition.

    Args:
        manifest: The results manifest for the model under test.
        condition: A key of `CONDITIONS`.

    Returns:
        dict[int, list[tuple[int, int]]]: Instance id -> flipped cells. Only
        reachable alterations are kept, since an unreachable one is rejected
        before it ever reaches the planner.

    Raises:
        KeyError: If the condition is unknown.
    """
    spec = CONDITIONS[condition]
    rows = manifest[manifest["method"] == spec["method"]]
    rows = rows[rows["budget"] == spec["budget"]]
    if "objective" in spec:
        rows = rows[rows["objective"] == spec["objective"]]
    if "flag" in spec and spec["flag"] in rows.columns:
        rows = rows[rows[spec["flag"]].to_numpy(dtype=bool, na_value=False)]
    rows = rows[rows["reachable"].astype(bool)]

    chosen: Dict[int, List[Tuple[int, int]]] = {}
    for instance_id, group in rows.groupby("instance_id"):
        # Parquet stores the cell list as an object array of length-2 arrays.
        cells = group.iloc[0]["flip_cells"]
        chosen[int(instance_id)] = [(int(cell[0]), int(cell[1])) for cell in cells]
    return chosen


def extend_alterations(
    instances: Sequence,
    alterations: Dict[int, List[Tuple[int, int]]],
    seed: int,
    polarity: str,
) -> Dict[int, List[Tuple[int, int]]]:
    """Add a second flip to each alteration, of the requested direction.

    Args:
        instances: The split's instances, keyed on by id.
        alterations: Instance id -> the single cell to block.
        seed: Seed for choosing the second cell.
        polarity: ``"free"`` or ``"block"``; see `paired_alteration`.

    Returns:
        dict[int, list[tuple[int, int]]]: The two-cell alterations. An instance
        is dropped when no second cell keeps the goal reachable.
    """
    by_id = {inst.instance_id: inst for inst in instances}
    rng = np.random.default_rng(seed)
    extended: Dict[int, List[Tuple[int, int]]] = {}
    for instance_id, cells in alterations.items():
        instance = by_id.get(instance_id)
        if instance is None or len(cells) != 1:
            continue
        paired = paired_alteration(instance, cells[0], rng, polarity=polarity)
        if paired is not None:
            extended[instance_id] = paired
    return extended


def make_feature_builder(group: str):
    """Build the callable that turns ``(instance, map_design)`` into surrogate input.

    The features are recomputed on the altered map, not reused from the clean
    one — that is the whole point: the surrogate must be asked what the cost
    map *should* look like given the geometry the planner was actually handed.

    Args:
        group: Feature group the surrogate was fitted on, a key of
            `FEATURE_GROUPS`. Path features are rejected: they are defined
            relative to a map's own optimum, so on an altered map they would
            quietly encode the alteration rather than describe the geometry.

    Returns:
        Callable: ``(instance, map_design) -> (matrix, mask)`` where `matrix`
        has one row per usable cell and `mask` marks those cells.

    Raises:
        ValueError: If the group contains a path feature.
    """
    names = FEATURE_GROUPS[group]
    if any(name.startswith("on_") for name in names):
        raise ValueError(
            f"group {group!r} contains path features, which are only defined "
            "relative to the clean map's optimum; use a geometry-only group"
        )

    def builder(instance, map_design: np.ndarray):
        """Compute the surrogate's features for one (possibly altered) map."""
        per_cell = compute_instance_features(replace(instance, map_design=map_design))
        mask = (per_cell["is_free"] > 0) & (per_cell["reachable"] > 0)
        matrix = np.stack([per_cell[name][mask] for name in names], axis=1).astype(np.float64)
        return matrix, mask

    return builder


def clean_residual_sd(
    planner, surrogate, instances: Sequence, builder, limit: int = 20
) -> float:
    """Residual spread on clean maps, used to scale the outlier-count feature.

    Args:
        planner: The planner under test.
        surrogate: The fitted surrogate.
        instances: Clean instances, normally from the validation split so the
            scale is not measured on anything the detector is scored on.
        builder: Feature builder from `make_feature_builder`.
        limit: How many instances to average over.

    Returns:
        float: Standard deviation of the residual over free, reachable cells.
    """
    import torch

    from src.planner import encode_cost_map

    values: List[np.ndarray] = []
    for instance in instances[:limit]:
        maps, starts, goals = instance.to_tensors()
        with torch.no_grad():
            cost = encode_cost_map(planner, maps, starts, goals)[0, 0].numpy()
        matrix, mask = builder(instance, instance.map_design)
        values.append(cost[mask] - surrogate.model.predict(matrix))
    return float(np.concatenate(values).std()) if values else 1.0


def main() -> int:
    """Fit and score both detectors on every requested condition."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--instances", type=int, default=None, help="cap, for probes")
    parser.add_argument(
        "--conditions", nargs="+", default=list(CONDITIONS), help="which alteration sets"
    )
    parser.add_argument(
        "--group", default="geometric_h1", help="surrogate feature group (geometry only)"
    )
    parser.add_argument("--kind", default="gbm", choices=["ridge", "gbm"])
    parser.add_argument(
        "--family", default="mazes", choices=list(config.MAP_FAMILIES), help="map family"
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--tag", default="", help="model tag; selects the tagged archives")
    parser.add_argument(
        "--transfer-from",
        default=None,
        help="train the detector on this condition and score it on all the others; "
        "PLAN.md asks for a detector fitted on k=4 optimality flips and tested across "
        "budgets and objectives, which this implements",
    )
    parser.add_argument("--timing-only", action="store_true")
    args = parser.parse_args()

    config.set_global_seeds(args.seed)
    planner = load_pretrained_neural_astar(checkpoint_path=args.checkpoint)
    suffix = f"_{args.tag}" if args.tag else ""

    train_matrix = build_design_matrix("train", tag=args.tag)
    test_matrix = build_design_matrix(args.split, tag=args.tag)
    surrogate = fit_and_score(train_matrix, test_matrix, args.group, kind=args.kind, seed=args.seed)
    print(f"surrogate [{args.group}/{args.kind}] R² = {surrogate.r2:.3f} on {args.split}")

    split = load_split(args.split, npz_path=config.family_npz(args.family))
    instances = split.instances(seed=args.seed)
    builder = make_feature_builder(args.group)

    valid = load_split("valid", npz_path=config.family_npz(args.family)).instances(seed=args.seed)
    spread = clean_residual_sd(planner, surrogate, valid, builder)
    print(f"clean residual sd = {spread:.4f} (validation split)")

    manifest = pd.read_parquet(config.RESULTS_DIR / f"manifest{suffix}.parquet")
    manifest = manifest[manifest["split"] == args.split]

    def examples_for(condition: str):
        """Build the detector's examples for one condition, or None if too few."""
        alterations = select_alterations(manifest, condition)
        polarity = CONDITIONS[condition].get("polarity")
        if polarity:
            alterations = extend_alterations(instances, alterations, args.seed, polarity)
        if args.instances is not None:
            keep = sorted(alterations)[: args.instances]
            alterations = {k: alterations[k] for k in keep}
        if len(alterations) < 8:
            print(f"{condition}: only {len(alterations)} instances, skipped")
            return None
        usable = [i for i in instances if i.instance_id in alterations]
        return build_examples(
            planner, surrogate, usable, alterations, builder, clean_sd=spread
        )

    # The transfer test fits one head on a single condition and never refits it.
    transfer_heads = None
    transfer_clean: Dict[str, np.ndarray] = {}
    source_first: set = set()
    if args.transfer_from:
        built = examples_for(args.transfer_from)
        if built is None:
            print(f"cannot train on {args.transfer_from}: too few instances")
            return 1
        source_residual, source_activations, source_plain, source_labels, source_ids = built
        source_unique = sorted(set(source_ids))
        source_first = set(source_unique[: len(source_unique) // 2])
        source_train = np.array([i in source_first for i in source_ids])
        # The clean example of a map is the same whatever condition is being
        # detected, so the out-of-distribution baseline is fitted once, on the
        # clean maps of the training half, and reused like the other heads.
        clean_source = source_train & (source_labels == 0)
        transfer_clean = {
            variant: matrix[clean_source] for variant, matrix in source_activations.items()
        }
        transfer_heads = {
            "residual": fit_detector(
                source_residual[source_train], source_labels[source_train]
            ),
            "map_only": fit_detector(source_plain[source_train], source_labels[source_train]),
            "combined": fit_detector(
                np.hstack([source_plain, source_residual])[source_train],
                source_labels[source_train],
            ),
        }
        print(
            f"detector trained on `{args.transfer_from}` "
            f"({len(source_first)} maps); every row below is scored by that one head, "
            "on the held-out maps only"
        )

    rows: List[Dict[str, object]] = []
    for condition in args.conditions:
        started = time.perf_counter()
        built = examples_for(condition)
        if built is None:
            continue
        residual, activations, plain, labels, ids = built
        seconds = time.perf_counter() - started

        # Split by instance, so no map contributes to both halves. Under a
        # transfer test the split is inherited from the condition the head was
        # trained on: the *same* maps must stay held out, because a map's clean
        # example is identical in every condition and the head has seen it.
        unique = sorted(set(ids))
        first_half = source_first if transfer_heads else set(unique[: len(unique) // 2])
        is_train = np.array([i in first_half for i in ids])
        # Guard against the leak this script was written with once: a map's
        # clean example is byte-identical in every condition, so scoring a
        # transferred head on a map it was fitted on inflates every number —
        # the Mahalanobis baseline went to a flat 1.000 before this check.
        assert not (source_first & set(np.asarray(ids)[~is_train].tolist())), (
            "transfer test is scoring maps the detector was trained on"
        )

        heads = transfer_heads or {}
        scores = evaluate_detector(
            residual[is_train],
            labels[is_train],
            residual[~is_train],
            labels[~is_train],
            model=heads.get("residual"),
        )
        control = evaluate_detector(
            plain[is_train],
            labels[is_train],
            plain[~is_train],
            labels[~is_train],
            model=heads.get("map_only"),
        )
        # The sharp question: does the residual carry anything the grid alone
        # does not? Only an improvement over the control here is evidence that
        # the planner's internals were informative.
        joint = np.hstack([plain, residual])
        combined = evaluate_detector(
            joint[is_train],
            labels[is_train],
            joint[~is_train],
            labels[~is_train],
            model=heads.get("combined"),
        )
        # Score the baseline at every layer and pooling, then report its best.
        # Giving it the pick of eight configurations against the residual
        # detector's one makes any win for the residual detector conservative.
        per_variant = {
            variant: evaluate_mahalanobis(
                transfer_clean[variant] if transfer_heads else matrix[is_train & (labels == 0)],
                matrix[~is_train],
                labels[~is_train],
            )["auc"]
            for variant, matrix in activations.items()
        }
        best_variant = max(per_variant, key=per_variant.get)
        baseline = {"auc": per_variant[best_variant], "variant": best_variant}
        # The harder, realistic version of the task: each evaluation map
        # contributes either its clean row or its altered row, never both, so
        # the detector cannot lean on a clean reference for the same map.
        keep = unpaired_mask(labels, is_train, ids)
        unpaired_ids = [identifier for identifier, k in zip(ids, keep) if k]
        unpaired = {
            name: evaluate_detector(
                matrix[keep][is_train[keep]],
                labels[keep][is_train[keep]],
                matrix[keep][~is_train[keep]],
                labels[keep][~is_train[keep]],
                model=heads.get(name),
            )["auc"]
            for name, matrix in (
                ("residual", residual),
                ("map_only", plain),
                ("combined", joint),
            )
        }
        unpaired_gap = bootstrap_auc_gap(
            joint[keep],
            plain[keep],
            labels[keep],
            is_train[keep],
            unpaired_ids,
            seed=args.seed,
            models=(heads.get("combined"), heads.get("map_only")),
        )

        gap = bootstrap_auc_gap(
            joint,
            plain,
            labels,
            is_train,
            ids,
            seed=args.seed,
            models=(heads.get("combined"), heads.get("map_only")),
        )
        rows.append(
            {
                "condition": condition,
                "n_instances": len(unique),
                "n_test_pairs": int((~is_train).sum() // 2),
                "residual_auc": scores["auc"],
                "residual_accuracy": scores["accuracy"],
                "map_only_auc": control["auc"],
                "map_only_accuracy": control["accuracy"],
                "combined_auc": combined["auc"],
                "combined_minus_map_only": gap,
                "unpaired": unpaired,
                "unpaired_combined_minus_map_only": unpaired_gap,
                "mahalanobis_auc": baseline["auc"],
                "mahalanobis_variant": baseline["variant"],
                "mahalanobis_auc_by_variant": per_variant,
                "seconds": seconds,
            }
        )
        print(
            f"{condition}: n={len(unique)} | residual AUC {scores['auc']:.3f} "
            f"(acc {scores['accuracy']:.3f}) | mahalanobis AUC {baseline['auc']:.3f} "
            f"[{baseline['variant']}] | map-only AUC {control['auc']:.3f} "
            f"| combined {combined['auc']:.3f} "
            f"(gap {gap['gap']:+.3f} [{gap['lo']:+.3f}, {gap['hi']:+.3f}]) "
            f"| unpaired: map-only {unpaired['map_only']:.3f}, "
            f"combined {unpaired['combined']:.3f} "
            f"(gap {unpaired_gap['gap']:+.3f}) "
            f"| {seconds:.1f} s"
        )
        if args.timing_only:
            break

    if args.timing_only or not rows:
        return 0

    payload = {
        "split": args.split,
        "seed": args.seed,
        "family": args.family,
        "model_tag": args.tag or "pretrained",
        "transfer_from": args.transfer_from,
        "surrogate": {"group": args.group, "kind": args.kind, "r2": surrogate.r2},
        "clean_residual_sd": spread,
        "residual_features": list(RESIDUAL_FEATURES),
        "map_only_features": list(MAP_ONLY_FEATURES),
        "conditions": rows,
    }
    transfer = f"_from_{args.transfer_from}" if args.transfer_from else ""
    out = config.RESULTS_DIR / f"e4_detector{suffix}{transfer}.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        f"# Step 12 — detecting altered maps ({args.tag or 'pretrained'}, {args.family})",
        "",
        f"Split `{args.split}`, seed {args.seed}. Surrogate: `{args.group}`/`{args.kind}`, "
        f"R² {surrogate.r2:.3f}. Clean residual sd {spread:.4f}.",
        "",
        (
            f"Detector trained on `{args.transfer_from}` and scored on every condition "
            "below without refitting — the transfer test `PLAN.md` asks for."
            if args.transfer_from
            else "Each detector is fitted on the first half of the maps and scored on the second."
        ),
        "",
        "Paired task: the detector sees each instance's clean and altered map and must "
        "separate them. AUC 0.5 is chance. The Mahalanobis baseline is scored at four "
        "encoder layers with two poolings each and reported at its best, so it is given "
        "every advantage. The map-only column is the control: five statistics of the grid "
        "itself, no model involved. A detector that does not beat it has learned nothing "
        "about the planner. The last column bootstraps the gap between the combined "
        "detector and that control over evaluation instances: an interval straddling zero "
        "means the residual added nothing measurable.",
        "",
        "| condition | instances | test pairs | residual AUC | residual accuracy "
        "| mahalanobis AUC | best layer/pooling | map-only AUC | map-only + residual "
        "| gap, 95 % CI |",
        "|-----------|-----------|------------|--------------|-------------------"
        "|-----------------|--------------------|--------------|---------------------"
        "|--------------|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['condition']}` | {row['n_instances']} | {row['n_test_pairs']} | "
            f"{row['residual_auc']:.3f} | {row['residual_accuracy']:.3f} | "
            f"{row['mahalanobis_auc']:.3f} | `{row['mahalanobis_variant']}` | "
            f"{row['map_only_auc']:.3f} | {row['combined_auc']:.3f} | "
            f"{row['combined_minus_map_only']['gap']:+.3f} "
            f"[{row['combined_minus_map_only']['lo']:+.3f}, "
            f"{row['combined_minus_map_only']['hi']:+.3f}] |"
        )
    lines.extend(
        [
            "",
            "## The harder version: no clean twin",
            "",
            "Each evaluation map contributes either its clean map or its altered one, "
            "never both, so the detector cannot compare a map against itself. This is "
            "what a deployed monitor faces, and these are the realistic numbers.",
            "",
            "| condition | residual AUC | map-only AUC | map-only + residual | gap, 95 % CI |",
            "|-----------|--------------|--------------|---------------------|--------------|",
        ]
    )
    for row in rows:
        values = row["unpaired"]
        moved = row["unpaired_combined_minus_map_only"]
        lines.append(
            f"| `{row['condition']}` | {values['residual']:.3f} | "
            f"{values['map_only']:.3f} | {values['combined']:.3f} | "
            f"{moved['gap']:+.3f} [{moved['lo']:+.3f}, {moved['hi']:+.3f}] |"
        )
    lines.append("")
    report = config.RESULTS_DIR / f"e4_detector{suffix}{transfer}.md"
    report.write_text("\n".join(lines) + "\n")
    print(f"  wrote {out.name} and {report.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
