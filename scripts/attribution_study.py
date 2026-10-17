"""PLAN.md step 8: attribution figures for a curated set of cells (E1).

Explains individual cost-map values with integrated gradients, on cells chosen
to contrast:

* **attacked vs clean** — cells the exhaustive sweep found damaging versus
  cells whose flip changes nothing;
* **wall-adjacent vs interior** — the partition that carries signal on this
  map family (the plan's junction/corridor split is degenerate here).

Also runs the randomisation sanity check of Adebayo et al. (2018) across the
whole curated set, and verifies that attribution mass stays inside the
encoder's receptive field — a cheap falsification test the architecture makes
available.

Usage:
    python scripts/attribution_study.py --per-group 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.data import load_split  # noqa: E402
from src.features import WALL_ZONE_ADJACENT, WALL_ZONE_INTERIOR, wall_zone  # noqa: E402
from src.attribution import (  # noqa: E402
    attribution_radius,
    grad_cam,
    integrated_gradients,
    randomization_check,
)
from src.planner import load_pretrained_neural_astar  # noqa: E402

GROUPS = ("attacked", "clean", "wall_adjacent", "interior")


def curate(
    split, manifest: Optional[pd.DataFrame], per_group: int, seed: int
) -> List[Dict[str, object]]:
    """Choose the cells to explain, balanced across the four contrasts.

    Args:
        split: The loaded `MazeSplit`.
        manifest: The step-6 manifest, or None. Without it the attacked and
            clean groups cannot be formed and are skipped.
        per_group: How many cells per group.
        seed: Seed for the random choices inside each group.

    Returns:
        list[dict]: Entries with ``group``, ``instance_id`` and ``cell``.
    """
    rng = np.random.default_rng(seed)
    chosen: List[Dict[str, object]] = []

    if manifest is not None:
        damaging = manifest[manifest["opt_C1O1"] > manifest["opt_C0O0"]]
        worst = damaging.sort_values("opt_C1O1", ascending=False).head(per_group)
        for _, row in worst.iterrows():
            chosen.append(
                {
                    "group": "attacked",
                    "instance_id": int(row["instance_id"]),
                    "cell": tuple(int(v) for v in row["flip_cells"][0]),
                    "opt_after_flip": float(row["opt_C1O1"]),
                }
            )
        harmless = manifest[
            (manifest["opt_C1O1"] <= manifest["opt_C0O0"])
            & (manifest["instance_id"].isin([c["instance_id"] for c in chosen]))
        ]
        for _, row in harmless.sample(n=min(per_group, len(harmless)), random_state=seed).iterrows():
            chosen.append(
                {
                    "group": "clean",
                    "instance_id": int(row["instance_id"]),
                    "cell": tuple(int(v) for v in row["flip_cells"][0]),
                    "opt_after_flip": float(row["opt_C1O1"]),
                }
            )

    for group, code in (("wall_adjacent", WALL_ZONE_ADJACENT), ("interior", WALL_ZONE_INTERIOR)):
        picked = 0
        for instance_id in rng.permutation(len(split)):
            if picked >= per_group:
                break
            instance = split.instance(int(instance_id), seed=seed)
            zones = wall_zone(instance.map_design)
            candidates = np.argwhere(zones == code)
            if not len(candidates):
                continue
            cell = candidates[rng.integers(len(candidates))]
            chosen.append(
                {
                    "group": group,
                    "instance_id": int(instance_id),
                    "cell": (int(cell[0]), int(cell[1])),
                }
            )
            picked += 1
    return chosen


def render_figure(planner, split, entries: List[Dict[str, object]], out_dir: Path, seed: int):
    """Draw map, attribution and Grad-CAM for each curated cell.

    Args:
        planner: The pretrained planner.
        split: The loaded `MazeSplit`.
        entries: Curated entries from `curate`.
        out_dir: Directory for the figures (one file per group).
        seed: Start-selection seed.

    Returns:
        list[Path]: The figures written.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    by_group: Dict[str, List[Dict[str, object]]] = {}
    for entry in entries:
        by_group.setdefault(str(entry["group"]), []).append(entry)

    for group, rows in by_group.items():
        fig, axes = plt.subplots(len(rows), 3, figsize=(7.6, 2.5 * len(rows)))
        axes = np.atleast_2d(axes)
        for r, entry in enumerate(rows):
            instance = split.instance(int(entry["instance_id"]), seed=seed)
            cell = tuple(entry["cell"])  # type: ignore[arg-type]
            ig = integrated_gradients(planner, instance, cell, steps=64)
            cam = grad_cam(planner, instance, cell)

            axes[r, 0].imshow(instance.map_design, cmap="gray", interpolation="nearest")
            axes[r, 0].plot(cell[1], cell[0], "rx", markersize=7)
            axes[r, 0].set_ylabel(f"inst {entry['instance_id']}", fontsize=8)
            values = ig.values[0]
            limit = max(float(np.abs(values).max()), 1e-9)
            axes[r, 1].imshow(
                values, cmap="coolwarm", vmin=-limit, vmax=limit, interpolation="nearest"
            )
            axes[r, 1].plot(cell[1], cell[0], "kx", markersize=6)
            axes[r, 2].imshow(cam.values[0], cmap="magma", interpolation="nearest")
            axes[r, 2].plot(cell[1], cell[0], "cx", markersize=6)
            if r == 0:
                axes[r, 0].set_title("map (x = explained cell)", fontsize=8)
                axes[r, 1].set_title("integrated gradients", fontsize=8)
                axes[r, 2].set_title("Grad-CAM", fontsize=8)
        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"E1 attribution — {group} cells", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        path = out_dir / f"attribution_{group}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(path)
    return written


def main() -> int:
    """Run the attribution study and write figures plus the summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--per-group", type=int, default=5)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument(
        "--manifest", type=Path, default=config.RESULTS_DIR / "manifest.parquet"
    )
    args = parser.parse_args()

    config.set_global_seeds(args.seed)
    split = load_split(config.canonical_split(args.split))
    planner = load_pretrained_neural_astar()
    manifest = pd.read_parquet(args.manifest) if args.manifest.exists() else None
    if manifest is None:
        print(f"note: {args.manifest} missing — attacked/clean groups skipped")

    entries = curate(split, manifest, args.per_group, args.seed)
    print(f"curated {len(entries)} cells across {len(set(e['group'] for e in entries))} groups")

    started = time.perf_counter()
    records = []
    for entry in entries:
        instance = split.instance(int(entry["instance_id"]), seed=args.seed)
        cell = tuple(entry["cell"])  # type: ignore[arg-type]
        ig = integrated_gradients(planner, instance, cell, steps=args.steps)
        radius = attribution_radius(ig)
        check = randomization_check(
            planner, instance, cell, method="integrated_gradients", seed=args.seed, steps=16
        )
        records.append(
            {
                **entry,
                "cell": list(cell),
                "cost_value": ig.output_value,
                "baseline_value": ig.baseline_output,
                "completeness_error": ig.completeness_error,
                "attr_total_map": ig.total_by_channel()["map_design"],
                "attr_abs_total": float(np.abs(ig.values[0]).sum()),
                "max_radius": radius["max_radius"],
                "share_within_5": radius["share_within_5"],
                "share_on_target": radius["share_on_target"],
                "randomization_pearson": check["pearson"],
                "randomization_spearman": check["spearman"],
                "randomization_abs_cosine": check["abs_cosine"],
            }
        )
    seconds = time.perf_counter() - started

    frame = pd.DataFrame(records)
    out_dir = config.RESULTS_DIR / "e1_attribution"
    figures = render_figure(planner, split, entries, out_dir, args.seed)

    grouped = (
        frame.groupby("group")[
            ["attr_abs_total", "share_on_target", "max_radius", "randomization_pearson"]
        ]
        .mean()
        .reset_index()
    )
    summary = {
        "n_cells": len(frame),
        "seconds": seconds,
        "steps": args.steps,
        "max_radius_over_all_cells": float(frame["max_radius"].max()),
        "min_share_within_5": float(frame["share_within_5"].min()),
        "mean_completeness_error": float(frame["completeness_error"].mean()),
        "randomization": {
            "mean_pearson": float(frame["randomization_pearson"].mean()),
            "max_abs_pearson": float(frame["randomization_pearson"].abs().max()),
            "mean_spearman": float(frame["randomization_spearman"].mean()),
            "mean_abs_cosine": float(frame["randomization_abs_cosine"].mean()),
        },
        "by_group": grouped.to_dict("records"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    frame.to_csv(out_dir / "cells.csv", index=False)
    (out_dir / "summary.md").write_text(render_markdown(summary, grouped, figures) + "\n")

    print(f"\nreceptive-field check: max radius {summary['max_radius_over_all_cells']:.0f} "
          f"(must be <= 5), min share within 5 = {summary['min_share_within_5']:.3f}")
    print(f"randomisation check: mean r = {summary['randomization']['mean_pearson']:+.3f}, "
          f"max |r| = {summary['randomization']['max_abs_pearson']:.3f}")
    print(grouped.to_string(index=False))
    print(f"\nwrote {len(figures)} figures and summary to {out_dir.relative_to(config.REPO_ROOT)}")
    return 0


def render_markdown(summary: Dict[str, object], grouped: pd.DataFrame, figures) -> str:
    """Render the attribution summary.

    Args:
        summary: Assembled summary values.
        grouped: Per-group means.
        figures: Figure paths.

    Returns:
        str: Markdown document.
    """
    randomization = summary["randomization"]  # type: ignore[index]
    lines = [
        "# Step 8 — attribution of single cost-map values",
        "",
        f"{summary['n_cells']} curated cells, integrated gradients with "
        f"{summary['steps']} steps ({summary['seconds']:.0f} s). Figures: "
        + ", ".join(f"`{p.name}`" for p in figures)
        + ".",
        "",
        "## Method checks",
        "",
        f"* **Receptive field.** Attribution mass must lie within radius 5 of the "
        f"explained cell, since the encoder is five 3×3 convolutions and cannot be "
        f"influenced from further away. Observed maximum radius over all cells: "
        f"**{summary['max_radius_over_all_cells']:.0f}**; minimum share of mass within "
        f"radius 5: **{summary['min_share_within_5']:.3f}**.",
        f"* **Completeness.** Integrated gradients should sum to the change in the "
        f"explained value between baseline and input; mean absolute error "
        f"**{summary['mean_completeness_error']:.2e}**.",
        f"* **Randomisation (Adebayo et al., 2018).** Re-running the attribution with "
        f"randomised encoder weights should change it. Mean correlation between the "
        f"trained and randomised attributions: **{randomization['mean_pearson']:+.3f}** "
        f"(max |r| {randomization['max_abs_pearson']:.3f}, mean Spearman "
        f"{randomization['mean_spearman']:+.3f}). The absolute-value cosine is higher "
        f"({randomization['mean_abs_cosine']:.3f}), which is expected: the receptive "
        f"field constrains *where* mass can land regardless of training.",
        "",
        "## By group",
        "",
        "| group | mean \\|attribution\\| | share on the explained cell | mean radius | randomisation r |",
        "|-------|---------------------|-----------------------------|-------------|-----------------|",
    ]
    for _, row in grouped.iterrows():
        lines.append(
            f"| {row['group']} | {row['attr_abs_total']:.4f} | {row['share_on_target']:.3f} | "
            f"{row['max_radius']:.1f} | {row['randomization_pearson']:+.3f} |"
        )
    lines += ["", "Per-cell values: `cells.csv`.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
