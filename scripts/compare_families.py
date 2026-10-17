"""PLAN.md step 7b (part a): the maze-trained planner on the other map families.

The step-3 and step-4 findings were measured on one map family — thin walls
separating open rooms — and three of them are geometric claims that should
behave differently elsewhere. The nested dataset submodule ships seven more
32x32 families, so testing those claims costs no download and no training.

What this script measures, per family:

1. **map structure** — obstacle fraction, how balanced the two cell partitions
   are, and how often a single flip can disconnect the goal;
2. **transfer** — Opt / Exp / Hmean of the maze-trained planner, which is a
   measure of *distribution shift*, not of whether the findings generalise;
3. **the pillar/door prediction** — on mazes, flipping a wall open ("door") was
   about five times more damaging than dropping an obstacle into open space
   ("pillar"). Families with no walls should show no such asymmetry.

The honest part (b) — training an encoder per family and repeating steps 3-7 —
needs a GPU and is tracked separately. Nothing here substitutes for it.

Usage:
    python scripts/compare_families.py --families mazes forest single_bugtrap
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.data import load_split  # noqa: E402
from src.exact_solver import solve_astar, solve_dijkstra  # noqa: E402
from src.features import (  # noqa: E402
    TOPOLOGY_LABELS,
    WALL_ZONE_LABELS,
    topology_class,
    wall_zone,
)
from src.grid import flip_cells, is_reachable  # noqa: E402
from src.metrics import (  # noqa: E402
    exp_ratio,
    harmonic_mean,
    n_expansions,
    opt_ratio,
    path_cost,
)
from src.perturb import (  # noqa: E402
    AlterationRequest,
    evaluate_alterations,
    sample_random_flips,
    summarize_by_direction,
)
from src.planner import load_pretrained_neural_astar  # noqa: E402


def map_statistics(split, n_sealing_probe: int = 5) -> Dict[str, object]:
    """Structural description of a map family.

    Args:
        split: A loaded `MazeSplit`.
        n_sealing_probe: How many instances to test exhaustively for cells
            whose removal disconnects the goal.

    Returns:
        dict: Obstacle fraction, the two partitions' class shares, and the
        measured rate of goal-disconnecting single flips.
    """
    obstacle_fraction = 1.0 - split.map_designs.reshape(len(split), -1).mean(axis=1)
    zones: Dict[str, float] = {}
    topology: Dict[str, float] = {}
    for index in range(min(len(split), 20)):
        design = split.map_designs[index]
        for code, label in WALL_ZONE_LABELS.items():
            zones[label] = zones.get(label, 0.0) + float((wall_zone(design) == code).mean())
        for code, label in TOPOLOGY_LABELS.items():
            topology[label] = topology.get(label, 0.0) + float(
                (topology_class(design) == code).mean()
            )
    divisor = min(len(split), 20)
    zones = {k: v / divisor for k, v in zones.items()}
    topology = {k: v / divisor for k, v in topology.items()}

    sealing, tried = 0, 0
    for index in range(min(len(split), n_sealing_probe)):
        instance = split.instance(index, seed=0)
        for cell in np.argwhere(instance.map_design == config.FREE):
            rc = (int(cell[0]), int(cell[1]))
            if rc in (instance.start_rc, instance.goal_rc):
                continue
            tried += 1
            if not is_reachable(
                flip_cells(instance.map_design, [rc]), instance.start_rc, instance.goal_rc
            ):
                sealing += 1
    return {
        "obstacle_fraction_mean": float(obstacle_fraction.mean()),
        "wall_zone_shares": zones,
        "topology_shares": topology,
        "sealing_flips_tried": tried,
        "sealing_flips_found": sealing,
        "sealing_rate": sealing / tried if tried else float("nan"),
    }


def transfer_metrics(planner, split, seed: int) -> Dict[str, float]:
    """Planning quality of the maze-trained planner on a family.

    Args:
        planner: The pretrained `NeuralAstar`.
        split: A loaded `MazeSplit`.
        seed: Start-selection seed.

    Returns:
        dict[str, float]: ``p_opt``, ``mean_opt``, ``mean_exp``, ``p_exp`` and
        ``h_mean``.
    """
    instances = split.instances(seed=seed)
    tensors = [inst.to_tensors() for inst in instances]
    maps = torch.cat([t[0] for t in tensors])
    starts = torch.cat([t[1] for t in tensors])
    goals = torch.cat([t[2] for t in tensors])
    with torch.no_grad():
        out = planner(maps, starts, goals)
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
        "p_exp": p_exp,
        "h_mean": harmonic_mean(p_opt, p_exp),
    }


def pillar_door_split(planner, split, seed: int, draws: int, instances_cap: int):
    """Damage rate of single flips, split by what the flip did to the map.

    Args:
        planner: The pretrained planner.
        split: A loaded `MazeSplit`.
        seed: Seed for the draws.
        draws: Random draws per instance.
        instances_cap: How many instances to use.

    Returns:
        list[dict]: Output of `src.perturb.summarize_by_direction`.
    """
    instances = split.instances(seed=seed)[:instances_cap]
    requests = []
    for instance in instances:
        for draw in range(draws):
            rng = np.random.default_rng([seed, instance.instance_id, 1, draw])
            cells = sample_random_flips(instance, 1, rng)
            if cells is not None:
                requests.append(
                    AlterationRequest(instance=instance, flip_cells=cells, draw=draw)
                )
    rows = evaluate_alterations(planner, requests, seed=seed)
    return summarize_by_direction(rows, budget=1)


def main() -> int:
    """Compare the families and write the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", nargs="+", default=list(config.MAP_FAMILIES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--draws", type=int, default=3)
    parser.add_argument("--instances", type=int, default=100)
    args = parser.parse_args()

    config.set_global_seeds(args.seed)
    planner = load_pretrained_neural_astar()
    report: Dict[str, object] = {"seed": args.seed, "families": {}}

    for family in args.families:
        started = time.perf_counter()
        split = load_split("test", npz_path=config.family_npz(family))
        entry = {
            "structure": map_statistics(split),
            "transfer": transfer_metrics(planner, split, args.seed),
            "pillar_door": pillar_door_split(
                planner, split, args.seed, args.draws, args.instances
            ),
        }
        entry["seconds"] = time.perf_counter() - started
        report["families"][family] = entry  # type: ignore[index]
        transfer = entry["transfer"]
        print(
            f"{family:18} obstacles {entry['structure']['obstacle_fraction_mean']:.3f}  "
            f"Opt {transfer['p_opt']:.2f}  Exp {transfer['mean_exp']:.2f}  "
            f"Hmean {transfer['h_mean']:.3f}  ({entry['seconds']:.0f} s)",
            flush=True,
        )

    (config.RESULTS_DIR / "family_comparison.json").write_text(
        json.dumps(report, indent=2, default=float) + "\n"
    )
    (config.RESULTS_DIR / "family_comparison.md").write_text(render_markdown(report) + "\n")
    print("\nwrote family_comparison.md and family_comparison.json")
    return 0


def render_markdown(report: Dict[str, object]) -> str:
    """Render the family comparison.

    Args:
        report: Assembled per-family results.

    Returns:
        str: Markdown document.
    """
    families: Dict[str, Dict[str, object]] = report["families"]  # type: ignore[assignment]
    lines = [
        "# Step 7b (part a) — the maze-trained planner across map families",
        "",
        "All eight 32x32 families ship with the nested dataset submodule. The planner "
        "here is the one trained on `mazes`, so every number below mixes *distribution "
        "shift* with *geometry*: it cannot show that a finding generalises, only that "
        "it does or does not survive the shift. Part (b) — one encoder trained per "
        "family — is what settles generalisation.",
        "",
        "## Map structure",
        "",
        "| family | obstacles | interior | wall-adjacent | open (topology) | dead ends | single flips that seal the goal |",
        "|--------|-----------|----------|---------------|-----------------|-----------|---------------------------------|",
    ]
    for name, entry in families.items():
        structure = entry["structure"]
        zones = structure["wall_zone_shares"]
        topology = structure["topology_shares"]
        lines.append(
            f"| `{name}` | {structure['obstacle_fraction_mean']:.1%} | "
            f"{zones.get('interior', 0):.0%} | {zones.get('wall_adjacent', 0):.0%} | "
            f"{topology.get('open', 0):.0%} | {topology.get('dead_end', 0):.1%} | "
            f"{structure['sealing_rate']:.2%} |"
        )
    lines += [
        "",
        "## Transfer of the maze-trained planner",
        "",
        "| family | exactly optimal | mean Opt | mean Exp | Hmean |",
        "|--------|-----------------|----------|----------|-------|",
    ]
    for name, entry in families.items():
        transfer = entry["transfer"]
        lines.append(
            f"| `{name}` | {transfer['p_opt']:.3f} | {transfer['mean_opt']:.4f} | "
            f"{transfer['mean_exp']:.3f} | {transfer['h_mean']:.3f} |"
        )
    lines += [
        "",
        "## The pillar/door prediction",
        "",
        "On mazes, opening a wall (a *door*) was about five times more damaging per "
        "flip than dropping an obstacle into open space (a *pillar*). If that is "
        "geometry rather than a property of the planner, families without walls "
        "should not show it.",
        "",
        "| family | pillar share of draws | pillar damage | door damage | ratio |",
        "|--------|----------------------|---------------|-------------|-------|",
    ]
    for name, entry in families.items():
        rows = {r["direction"]: r for r in entry["pillar_door"]}
        pillar = rows.get("free->obstacle (pillar)")
        door = rows.get("obstacle->free (door)")
        if not pillar or not door:
            continue
        ratio = (
            door["frac_newly_suboptimal"] / pillar["frac_newly_suboptimal"]
            if pillar["frac_newly_suboptimal"]
            else float("inf")
        )
        lines.append(
            f"| `{name}` | {pillar['share_of_draws']:.0%} | "
            f"{pillar['frac_newly_suboptimal']:.1%} | {door['frac_newly_suboptimal']:.1%} | "
            f"{ratio:.1f}x |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
