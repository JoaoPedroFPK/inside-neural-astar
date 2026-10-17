"""PLAN.md step 5: fit surrogate models of the cost map and test sufficiency.

Fits the three variants the plan asks for (geometric-only, path-only, both) in
two model classes, reports residuals grouped by cell class, and then runs the
test R² cannot perform: it feeds the surrogate's predicted cost map to the real
search module and measures whether planning quality survives.

Writes `results/e1_surrogate.md`, `results/e1_surrogate.json` and
`results/e1_surrogate_residuals.png`.

Usage:
    python scripts/fit_surrogate.py                # all fits + sufficiency test
    python scripts/fit_surrogate.py --no-sufficiency
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
from src.exact_solver import solve_astar, solve_dijkstra  # noqa: E402
from src.metrics import (  # noqa: E402
    exp_ratio,
    harmonic_mean,
    n_expansions,
    opt_ratio,
    path_cost,
)
from src.planner import load_pretrained_neural_astar, run_astar_module  # noqa: E402
from src.surrogate import (  # noqa: E402
    FEATURE_GROUPS,
    build_design_matrix,
    fit_and_score,
    load_archive,
    residual_by_class,
    sufficiency_cost_maps,
    summarize_fits,
)

#: Fits reported in the headline table, in reading order.
FITS = [
    ("goal_to", "ridge"),
    ("heuristic", "ridge"),
    ("obstacle", "ridge"),
    ("geometric_h1", "ridge"),
    ("geometric", "ridge"),
    ("path", "ridge"),
    ("all", "ridge"),
    ("goal_to", "gbm"),
    ("goal", "gbm"),
    ("obstacle", "gbm"),
    ("geometric_h1", "gbm"),
    ("geometric", "gbm"),
    ("path", "gbm"),
    ("all", "gbm"),
]


def plan_with_cost_maps(planner, instances, cost_maps: np.ndarray) -> Dict[str, float]:
    """Run the search on given cost maps and score the result.

    Args:
        planner: A `NeuralAstar` whose `.astar` module is used; its encoder is
            bypassed entirely.
        instances: The `MazeInstance` objects, in archive order.
        cost_maps: ``(N, H, W)`` cost maps to search with.

    Returns:
        dict[str, float]: ``p_opt`` (share exactly optimal), ``mean_opt``,
        ``mean_exp`` and ``p_exp`` (the paper's floored reduction).
    """
    tensors = [inst.to_tensors() for inst in instances]
    maps = torch.cat([t[0] for t in tensors])
    starts = torch.cat([t[1] for t in tensors])
    goals = torch.cat([t[2] for t in tensors])
    costs = torch.from_numpy(cost_maps.astype(np.float32)).unsqueeze(1)

    with torch.no_grad():
        out = run_astar_module(planner, costs, starts, goals, maps)

    optimum, classical = [], []
    for inst in instances:
        optimum.append(solve_dijkstra(inst.map_design, inst.start_rc, inst.goal_rc).cost)
        classical.append(solve_astar(inst.map_design, inst.start_rc, inst.goal_rc).expansions)
    ratios_opt = opt_ratio(path_cost(out.paths), np.array(optimum))
    ratios_exp = exp_ratio(n_expansions(out.histories), np.array(classical, dtype=np.float64))
    p_opt = float(np.mean(ratios_opt <= 1.0))
    p_exp = float(np.nanmean(np.maximum(1.0 - ratios_exp, 0.0)))
    return {
        "p_opt": p_opt,
        "mean_opt": float(np.nanmean(ratios_opt)),
        "mean_exp": float(np.nanmean(ratios_exp)),
        "p_exp": p_exp,
        "h_mean": harmonic_mean(p_opt, p_exp),
    }


def render_residual_figure(archive, results, out_path: Path, rows=(0, 1, 2)) -> Path:
    """Draw cost map, surrogate prediction and residual for a few instances.

    Args:
        archive: The test feature archive.
        results: Mapping of label -> ``(N, H, W)`` predicted cost maps.
        out_path: Where to write the PNG.
        rows: Archive row indices to draw.

    Returns:
        Path: `out_path`.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(results)
    n_cols = 1 + 2 * len(labels)
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(2.1 * n_cols, 2.4 * len(rows)))
    axes = np.atleast_2d(axes)
    mask_all = (archive["is_free"] > 0) & (archive["reachable"] > 0)

    for r, index in enumerate(rows):
        usable = mask_all[index]
        truth = np.where(usable, archive["cost_map"][index], np.nan)
        axes[r, 0].imshow(truth, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
        if r == 0:
            axes[r, 0].set_title("cost map (encoder)", fontsize=8)
        axes[r, 0].set_ylabel(f"instance {archive['instance_ids'][index]}", fontsize=8)
        for k, label in enumerate(labels):
            prediction = np.where(usable, results[label][index], np.nan)
            axes[r, 1 + 2 * k].imshow(
                prediction, cmap="viridis", vmin=0, vmax=1, interpolation="nearest"
            )
            residual = np.where(usable, archive["cost_map"][index] - results[label][index], np.nan)
            axes[r, 2 + 2 * k].imshow(
                residual, cmap="coolwarm", vmin=-0.6, vmax=0.6, interpolation="nearest"
            )
            if r == 0:
                axes[r, 1 + 2 * k].set_title(f"{label}: prediction", fontsize=8)
                axes[r, 2 + 2 * k].set_title(f"{label}: residual", fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "E1 surrogate — encoder cost map, surrogate prediction, and what is left over "
        "(red = surrogate under-predicts)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def main() -> int:
    """Run every fit, the residual tables and the sufficiency test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-sufficiency", action="store_true")
    args = parser.parse_args()

    config.set_global_seeds(args.seed)
    train = build_design_matrix("train")
    test = build_design_matrix("test")
    archive = load_archive("test")
    print(f"train rows {len(train.y):,}  test rows {len(test.y):,}\n")

    started = time.perf_counter()
    fitted = {}
    for group, kind in FITS:
        fitted[(group, kind)] = fit_and_score(train, test, group, kind, seed=args.seed)
        r = fitted[(group, kind)]
        print(f"  {group:10} {kind:5}  R2={r.r2:6.3f}  within={r.r2_within:6.3f}")
    print(f"\nfits took {time.perf_counter() - started:.0f} s")

    best_geo = fitted[("geometric_h1", "gbm")]
    best_all = fitted[("all", "gbm")]
    report: Dict[str, object] = {
        "seed": args.seed,
        "n_train_rows": int(len(train.y)),
        "n_test_rows": int(len(test.y)),
        "fits": summarize_fits([fitted[k] for k in FITS]),
        "ridge_coefficients_all": fitted[("all", "ridge")].coefficients,
        "residuals_by_wall_zone": residual_by_class(best_all, test, "wall_zone"),
        "residuals_by_topology": residual_by_class(best_all, test, "topology_class"),
    }

    predictions = {
        "geometric_h1": sufficiency_cost_maps(best_geo, archive),
        "all": sufficiency_cost_maps(best_all, archive),
    }
    figure = render_residual_figure(
        archive, predictions, config.RESULTS_DIR / "e1_surrogate_residuals.png"
    )

    if not args.no_sufficiency:
        print("\nsufficiency test: planning with surrogate cost maps")
        planner = load_pretrained_neural_astar()
        split = load_split("test")
        instances = [split.instance(int(i), seed=args.seed) for i in archive["instance_ids"]]
        sufficiency = {
            "neural_encoder": plan_with_cost_maps(
                planner, instances, archive["cost_map"].astype(np.float32)
            ),
            "surrogate_geometric_h1": plan_with_cost_maps(
                planner, instances, predictions["geometric_h1"]
            ),
            "surrogate_all": plan_with_cost_maps(planner, instances, predictions["all"]),
            "classical_unit_cost": plan_with_cost_maps(
                planner, instances, np.stack([i.map_design for i in instances])
            ),
        }
        report["sufficiency"] = sufficiency
        for label, row in sufficiency.items():
            print(
                f"  {label:22} Opt={row['p_opt']:.3f}  meanOpt={row['mean_opt']:.4f}  "
                f"Exp={row['mean_exp']:.3f}  p_exp={row['p_exp']:.4f}  "
                f"Hmean={row['h_mean']:.4f}"
            )

    out_json = config.RESULTS_DIR / "e1_surrogate.json"
    out_json.write_text(json.dumps(report, indent=2, default=float) + "\n")
    (config.RESULTS_DIR / "e1_surrogate.md").write_text(render_markdown(report, figure) + "\n")
    print(f"\nwrote {out_json.relative_to(config.REPO_ROOT)}, e1_surrogate.md and {figure.name}")
    return 0


def render_markdown(report: Dict[str, object], figure: Path) -> str:
    """Render the step-5 report.

    Args:
        report: The assembled results.
        figure: Path to the residual figure.

    Returns:
        str: Markdown document.
    """
    fits: List[Dict[str, object]] = report["fits"]  # type: ignore[assignment]
    by_key = {(f["group"], f["kind"]): f for f in fits}
    lines = [
        "# Step 5 — surrogate fits of the cost map",
        "",
        f"Fitted on the train split ({report['n_train_rows']:,} free reachable cells of "
        f"800 maps), scored on the test split ({report['n_test_rows']:,} cells of 100 maps). "
        "R² is the share of the cost map's variance explained; the within-instance column "
        "removes each map's mean first, so a fit cannot score well merely by separating "
        "easy maps from hard ones.",
        "",
        "| feature group | n | ridge R² | ridge within | GBM R² | GBM within |",
        "|---------------|---|----------|--------------|--------|------------|",
    ]
    for group in (
        "goal_to",
        "heuristic",
        "obstacle",
        "geometric_h1",
        "geometric",
        "path",
        "all",
    ):
        ridge = by_key.get((group, "ridge"))
        gbm = by_key.get((group, "gbm"))
        if ridge is None:
            continue
        gbm_r2 = f"{gbm['r2']:.3f}" if gbm else "—"
        gbm_w = f"{gbm['r2_within']:.3f}" if gbm else "—"
        lines.append(
            f"| `{group}` | {ridge['n_features']} | {ridge['r2']:.3f} | "
            f"{ridge['r2_within']:.3f} | {gbm_r2} | {gbm_w} |"
        )
    lines += [
        "",
        "`goal_to` = the three distances to the goal; `heuristic` = the two "
        "features spanning the search's own heuristic; `obstacle` = distance "
        "transforms, densities and the two width proxies; **`geometric_h1` = "
        "`goal_to` + `obstacle`, which is hypothesis H1 exactly as stated**; "
        "`geometric` = the same plus `geodesic_from_start`; `path` = the three "
        "path indicators; `all` = everything.",
        "",
        "**Why `geometric` is not a clean test of H1.** A cell lies on some optimal "
        "path exactly when `geodesic_from_start + geodesic_to_goal == optimum`, so a "
        "non-linear model handed both distances can rebuild the path indicator "
        "without being given it. The gap between `geometric_h1` and `geometric` is "
        "that leakage, and it is large — which is itself informative about what the "
        "encoder encodes.",
        "",
        "## Residuals of the best fit (`all`, gradient boosting)",
        "",
        "Grouped by distance to the nearest wall. PLAN.md asks for the "
        "free-neighbour topology partition, which is degenerate on this map "
        "family, so both are reported.",
        "",
        "| wall zone | share of cells | mean \\|residual\\| | mean signed residual | mean cost |",
        "|-----------|----------------|-------------------|----------------------|-----------|",
    ]
    for row in report["residuals_by_wall_zone"]:  # type: ignore[index]
        lines.append(
            f"| {row['class']} | {row['share']:.1%} | {row['mean_abs_residual']:.4f} | "
            f"{row['mean_signed_residual']:+.4f} | {row['mean_cost']:.3f} |"
        )
    lines += [
        "",
        "| topology class | share of cells | mean \\|residual\\| | mean signed residual | mean cost |",
        "|----------------|----------------|-------------------|----------------------|-----------|",
    ]
    for row in report["residuals_by_topology"]:  # type: ignore[index]
        lines.append(
            f"| {row['class']} | {row['share']:.1%} | {row['mean_abs_residual']:.4f} | "
            f"{row['mean_signed_residual']:+.4f} | {row['mean_cost']:.3f} |"
        )
    lines += ["", f"Residual maps: `{figure.name}`.", ""]

    if "sufficiency" in report:
        lines += [
            "## Sufficiency test — can a surrogate cost map plan?",
            "",
            "R² measures reconstruction, not behaviour: a cost-map difference only "
            "matters if it changes where the search goes. Here the surrogate's "
            "prediction is assembled back into a cost map and handed to the *same* "
            "search module the encoder feeds. `classical_unit_cost` is the same "
            "module with a flat cost map, i.e. ordinary A\\*.",
            "",
            "| cost map | Opt (exactly optimal) | mean Opt ratio | mean Exp ratio | Hmean |",
            "|----------|----------------------|----------------|----------------|-------|",
        ]
        for label, row in report["sufficiency"].items():  # type: ignore[index]
            lines.append(
                f"| `{label}` | {row['p_opt']:.3f} | {row['mean_opt']:.4f} | "
                f"{row['mean_exp']:.3f} | {row['h_mean']:.4f} |"
            )
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
