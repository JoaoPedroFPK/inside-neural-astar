"""PLAN.md step 7: run the 2x2 swap on the damaging k=1 flips (E3).

Reads the manifest written by `scripts/sweep_k1.py`, takes the alterations
tagged as worst under each objective, and evaluates each one under all four
cost/obstacle conditions. Computes the mediation fractions, merges the swap
columns back into the manifest, and writes the report and the plot.

This is the project's first headline result: of the damage a single flipped
cell does, how much is reproduced by the encoder's distorted cost map alone,
and how much by the search reacting to the changed map alone.

Usage:
    python scripts/swap_k1.py                     # both objectives
    python scripts/swap_k1.py --objective optimality
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.data import load_split  # noqa: E402
from src.planner import load_pretrained_neural_astar  # noqa: E402
from src.swap import SwapResult, run_swap, summarize_swaps, swap_columns  # noqa: E402

#: Which manifest tag selects each objective, and which metric it is judged on.
OBJECTIVES = {
    "optimality": ("is_worst_optimality", "opt"),
    "efficiency": ("is_worst_efficiency", "exp"),
}


def render_plot(
    results: Dict[str, List[SwapResult]], out_path: Path
) -> Path:
    """Scatter the two mediation fractions, one panel per objective.

    Args:
        results: Objective name -> swap results.
        out_path: Where to write the PNG.

    Returns:
        Path: `out_path`.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(results), figsize=(5.4 * len(results), 5.0))
    axes = np.atleast_1d(axes)
    for ax, (objective, rows) in zip(axes, results.items()):
        metric = OBJECTIVES[objective][1]
        groups = {"optimum unchanged": ([], []), "optimum changed": ([], [])}
        for result in rows:
            fractions = result.mediation(metric)
            if not np.isfinite(fractions["total_effect"]) or fractions["total_effect"] <= 0:
                continue
            key = "optimum changed" if result.optimum_changed else "optimum unchanged"
            groups[key][0].append(fractions["encoder"])
            groups[key][1].append(fractions["search"])
        ax.axhline(0, color="0.8", lw=0.8)
        ax.axvline(0, color="0.8", lw=0.8)
        ax.axhline(1, color="0.9", lw=0.8, ls="--")
        ax.axvline(1, color="0.9", lw=0.8, ls="--")
        ax.plot([-0.5, 1.5], [1.5, -0.5], color="0.85", lw=0.8, ls=":")
        total = 0
        for (label, (encoder, search)), marker in zip(groups.items(), ("o", "^")):
            if not encoder:
                continue
            total += len(encoder)
            ax.scatter(
                encoder, search, s=26, alpha=0.6, marker=marker, edgecolor="none",
                label=f"{label} (n={len(encoder)})",
            )
        ax.legend(fontsize=8, loc="upper right")
        ax.set_xlabel("encoder route  (C1O0 share of the damage)")
        ax.set_ylabel("search route  (C0O1 share of the damage)")
        ax.set_title(
            f"{objective} objective — metric {metric.upper()}  (n={total})", fontsize=10
        )
        ax.set_xlim(-0.5, 1.6)
        ax.set_ylim(-0.5, 1.6)
        ax.set_aspect("equal")
    fig.suptitle(
        "E3 — mediation of single-flip damage: encoder route vs search route\n"
        "bottom-left = interaction (neither alone reproduces it); "
        "dotted line = the two routes summing to the full effect",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def render_markdown(
    summaries: Dict[str, Dict[str, object]],
    results: Dict[str, List[SwapResult]],
    meta: Dict[str, object],
    figure: Path,
    summaries_raw: Optional[Dict[str, Dict[str, object]]] = None,
    summaries_split: Optional[Dict[str, Dict[str, Dict[str, object]]]] = None,
) -> str:
    """Render the step-7 report, including the one-paragraph reading.

    Args:
        summaries: Objective -> `summarize_swaps` output.
        results: Objective -> swap results.
        meta: Run metadata.
        figure: Path to the plot.
        summaries_raw: The same decomposition on raw counts, as a robustness
            check against the metric denominators differing between conditions.
        summaries_split: The decomposition split by whether the flip moved the
            map's optimum — two qualitatively different failure modes.

    Returns:
        str: Markdown document.
    """
    lines = [
        "# Step 7 — mediation of single-flip damage (E3)",
        "",
        f"Split `{meta['split']}`, seed {meta['seed']}. The alterations are the worst "
        "single-cell flips found by the exhaustive sweep of step 6 — one per map per "
        f"objective — evaluated under all four cost/obstacle conditions "
        f"({meta['n_swaps']} swaps, {meta['seconds']:.0f} s).",
        "",
        "Each condition is scored against the optimum and the classical-A\\* effort of "
        "**its own obstacle mask**, since the mask defines the problem actually being "
        "solved.",
        "",
    ]
    for objective, summary in summaries.items():
        metric = OBJECTIVES[objective][1]
        labels = summary["labels"]
        total = summary["n"]
        lines += [
            f"## {objective.capitalize()} objective (metric `{metric.upper()}`)",
            "",
            f"{total} alterations; {summary['n_with_damage']} of them actually changed "
            f"`{metric.upper()}` and so can be decomposed.",
            "",
            "| route pattern | alterations | share |",
            "|---------------|-------------|-------|",
        ]
        for label in ("encoder", "search", "both", "interaction", "no damage"):
            count = labels.get(label, 0)
            lines.append(f"| {label} | {count} | {count / total:.0%} |")
        lines += [
            "",
            f"Mediation fractions over the {summary['n_with_damage']} damaged "
            "alterations. The distribution is bimodal — an individual failure is "
            "usually reproduced almost entirely by one route or by neither — so the "
            "share at each end says more than the average:",
            "",
            "| route | mean | median | IQR | explains it alone (≥ 0.75) | contributes nothing (≤ 0.25) |",
            "|-------|------|--------|-----|----------------------------|------------------------------|",
            f"| encoder | {summary['encoder_mean']:+.3f} | {summary['encoder_median']:+.2f} | "
            f"{summary['encoder_q25']:+.2f} – {summary['encoder_q75']:+.2f} | "
            f"{summary['encoder_share_dominant']:.0%} | {summary['encoder_share_absent']:.0%} |",
            f"| search | {summary['search_mean']:+.3f} | {summary['search_median']:+.2f} | "
            f"{summary['search_q25']:+.2f} – {summary['search_q75']:+.2f} | "
            f"{summary['search_share_dominant']:.0%} | {summary['search_share_absent']:.0%} |",
            "",
            f"The two routes sum to {summary['route_sum_mean']:.3f} on average "
            f"(median {summary['route_sum_median']:.3f}). A sum near 1 means the "
            "decomposition is nearly additive — the two routes partition the damage; "
            "a sum well below 1 means the rest is interaction, needing both changes "
            "at once.",
            "",
        ]
        if summaries_split and objective in summaries_split:
            lines += [
                "### Split by failure mode",
                "",
                "Two different things are being called damage here. When the flip "
                "leaves the map's optimum unchanged, the planner took a worse path on "
                "a problem whose answer did not move. When the flip opens a new "
                "shortcut the optimum drops, and the planner is penalised for not "
                "using a route that exists only under the altered mask — the mask "
                "being exactly what the search route varies, so a search-dominated "
                "reading there is close to definitional. Pooling the two is "
                "misleading:",
                "",
                "| failure mode | n | encoder mean | encoder dominant | search mean | search dominant |",
                "|--------------|---|--------------|------------------|-------------|-----------------|",
            ]
            for mode, part in summaries_split[objective].items():
                if not part["n_with_damage"]:
                    continue
                lines.append(
                    f"| {mode} | {part['n_with_damage']} | {part['encoder_mean']:+.3f} | "
                    f"{part['encoder_share_dominant']:.0%} | {part['search_mean']:+.3f} | "
                    f"{part['search_share_dominant']:.0%} |"
                )
            lines.append("")
        if summaries_raw and objective in summaries_raw:
            raw = summaries_raw[objective]
            lines += [
                "Robustness check on raw counts (path length in moves, or expanded "
                "cells) rather than the ratios, which have a per-condition "
                f"denominator: encoder mean {raw['encoder_mean']:+.3f}, search mean "
                f"{raw['search_mean']:+.3f}, over {raw['n_with_damage']} alterations. "
                "The two need not agree exactly — when a flip changes the optimum, a "
                "ratio can worsen while the raw path length improves.",
                "",
            ]
    lines += [
        f"Plot: `{figure.name}`. Each point is one alteration; the axes are the two "
        "mediation fractions. A point near (1, 0) is encoder-mediated, near (0, 1) is "
        "search-mediated, near the origin means neither route reproduces the damage on "
        "its own.",
        "",
        "## Caveats",
        "",
        "* The off-diagonal conditions are off-distribution by construction: the "
        "encoder never produced that cost map for that obstacle mask. A strange "
        "`C1O0` can be the mismatch rather than the encoder's distortion.",
        "* The route labels use a tolerance of 0.25 around 0 and 1; they summarise the "
        "scatter, they are not a test.",
        "* These are the *worst* flips per map, selected by the same metric they are "
        "then decomposed on, so they are not a random sample of alterations.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    """Run the swap on the tagged flips and write the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--objective", default="both", choices=["both", "optimality", "efficiency"]
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="encoder checkpoint; must match the one the manifest was swept with",
    )
    parser.add_argument("--tag", default="", help="suffix for the manifest and reports")
    parser.add_argument(
        "--family",
        default="mazes",
        choices=list(config.MAP_FAMILIES),
        help="map family; all eight ship with the nested dataset submodule",
    )

    parser.add_argument(
        "--manifest", type=Path, default=None, help="derived from --tag if omitted"
    )
    args = parser.parse_args()

    suffix = f"_{args.tag}" if args.tag else ""
    if args.manifest is None:
        args.manifest = config.RESULTS_DIR / f"manifest{suffix}.parquet"
    if not args.manifest.exists():
        print(f"{args.manifest} not found; run scripts/sweep_k1.py first", file=sys.stderr)
        return 1

    split_name = config.canonical_split(args.split)
    config.set_global_seeds(args.seed)
    frame = pd.read_parquet(args.manifest)
    split = load_split(split_name, npz_path=config.family_npz(args.family))
    planner = load_pretrained_neural_astar(checkpoint_path=args.checkpoint)

    objectives = (
        list(OBJECTIVES) if args.objective == "both" else [args.objective]
    )
    results: Dict[str, List[SwapResult]] = {}
    swap_rows: Dict[int, Dict[str, object]] = {}
    started = time.perf_counter()

    for objective in objectives:
        tag, _ = OBJECTIVES[objective]
        selected = frame.index[frame[tag].fillna(False).astype(bool)].tolist()
        print(f"{objective}: {len(selected)} tagged alterations")
        rows: List[SwapResult] = []
        for position, index in enumerate(selected, start=1):
            record = frame.loc[index]
            instance = split.instance(int(record["instance_id"]), seed=args.seed)
            cells = [tuple(int(v) for v in cell) for cell in record["flip_cells"]]
            result = run_swap(planner, instance, cells)
            rows.append(result)
            swap_rows[index] = swap_columns(result)
            if position % 25 == 0:
                print(f"  {position}/{len(selected)}", flush=True)
        results[objective] = rows

    seconds = time.perf_counter() - started
    summaries = {
        objective: summarize_swaps(rows, OBJECTIVES[objective][1])
        for objective, rows in results.items()
    }
    summaries_raw = {
        objective: summarize_swaps(rows, OBJECTIVES[objective][1], raw=True)
        for objective, rows in results.items()
    }
    # The same decomposition split by whether the flip moved the map's optimum.
    # Pooling the two hides a qualitative difference; see the report.
    summaries_split = {
        objective: {
            "optimum unchanged": summarize_swaps(
                [r for r in rows if not r.optimum_changed], OBJECTIVES[objective][1]
            ),
            "optimum changed": summarize_swaps(
                [r for r in rows if r.optimum_changed], OBJECTIVES[objective][1]
            ),
        }
        for objective, rows in results.items()
    }

    # Merge the swap columns into the manifest, leaving untouched rows as NaN.
    additions = pd.DataFrame.from_dict(swap_rows, orient="index")
    for column in additions.columns:
        # Create the column with the addition's own dtype rather than float64,
        # so boolean validity flags are not silently coerced.
        if column not in frame.columns:
            empty = pd.Series(index=frame.index, dtype=additions[column].dtype)
            frame[column] = empty
        elif frame[column].dtype != additions[column].dtype:
            frame[column] = frame[column].astype(object)
        frame.loc[additions.index, column] = additions[column]
    frame.to_parquet(args.manifest, index=False)

    figure = render_plot(results, config.RESULTS_DIR / f"e3_k1_mediation{suffix}.png")
    meta = {
        "split": split_name,
        "seed": args.seed,
        "n_swaps": sum(len(r) for r in results.values()),
        "seconds": seconds,
    }
    (config.RESULTS_DIR / f"e3_k1{suffix}.json").write_text(
        json.dumps(
            {
                "meta": meta,
                "summaries": summaries,
                "summaries_raw": summaries_raw,
                "summaries_by_failure_mode": summaries_split,
            },
            indent=2,
            default=float,
        )
        + "\n"
    )
    (config.RESULTS_DIR / f"e3_k1{suffix}.md").write_text(
        render_markdown(summaries, results, meta, figure, summaries_raw, summaries_split)
        + "\n"
    )

    print()
    for objective, summary in summaries.items():
        print(
            f"{objective:11} encoder mean {summary['encoder_mean']:+.3f} "
            f"(dominant in {summary['encoder_share_dominant']:.0%})  "
            f"search mean {summary['search_mean']:+.3f} "
            f"(dominant in {summary['search_share_dominant']:.0%})"
        )
    print(f"\nwrote e3_k1.md, e3_k1.json, {figure.name}, and merged swap columns into the manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
