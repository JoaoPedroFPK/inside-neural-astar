"""PLAN.md step 0 (E0): reproduce the upstream maze numbers on the test split.

Runs the pretrained Neural A* and classical A* over a whole split and reports
the paper's three metrics (`p_opt`, `p_exp`, `h_mean`, see `src.metrics`) under
two start-selection protocols:

``upstream``
    Starts sampled by `neural_astar.utils.data.MazeDataset` exactly as in
    training/validation (random, from the 55-100th distance percentile band),
    with the global seed fixed. This is the protocol the published numbers come
    from, so it is the one to compare against the paper.

``fixed``
    Our reproducible starts (`src.data.choose_start`), one per instance, stored
    with every result. Every later experiment in this project uses these, so
    E0 has to be measured on them too — otherwise the perturbation baselines
    would not be comparable to the E0 line.

Usage:
    python scripts/e0_reproduce.py --split test --seed 0

Writes `results/e0_reproduce.json` (and prints a markdown table) unless
``--out`` is given.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.data import load_split  # noqa: E402
from src.metrics import (  # noqa: E402
    harmonic_mean,
    n_expansions,
    p_exp_reduction,
    p_opt_binary,
    path_cost,
)
from src.planner import load_pretrained_neural_astar, make_vanilla_astar  # noqa: E402


def evaluate_batches(planner, vanilla, batches) -> Dict[str, float]:
    """Run both planners over pre-built batches and aggregate the metrics.

    Args:
        planner: The Neural A* planner under test.
        vanilla: The classical A* control (same search module, unit costs).
        batches: Iterable of ``(map_designs, start_maps, goal_maps)`` tuples of
            ``(B, 1, H, W)`` tensors.

    Returns:
        dict[str, float]: ``p_opt``, ``p_exp``, ``h_mean``, the mean expansion
        counts of both planners, the mean path costs, and ``n`` (instances).
    """
    opt_flags, exp_reductions = [], []
    exp_neural, exp_vanilla, cost_neural, cost_vanilla = [], [], [], []

    with torch.no_grad():
        for map_designs, start_maps, goal_maps in batches:
            na = planner(map_designs, start_maps, goal_maps)
            va = vanilla(map_designs, start_maps, goal_maps)
            opt_flags.append(p_opt_binary(va.paths, na.paths))
            exp_reductions.append(p_exp_reduction(va.histories, na.histories))
            exp_neural.append(n_expansions(na.histories))
            exp_vanilla.append(n_expansions(va.histories))
            cost_neural.append(path_cost(na.paths))
            cost_vanilla.append(path_cost(va.paths))

    p_opt = float(np.concatenate(opt_flags).mean())
    p_exp = float(np.concatenate(exp_reductions).mean())
    return {
        "n": int(np.concatenate(opt_flags).size),
        "p_opt": p_opt,
        "p_exp": p_exp,
        "h_mean": harmonic_mean(p_opt, p_exp),
        "mean_expansions_neural": float(np.concatenate(exp_neural).mean()),
        "mean_expansions_classical": float(np.concatenate(exp_vanilla).mean()),
        "mean_path_cost_neural": float(np.concatenate(cost_neural).mean()),
        "mean_path_cost_classical": float(np.concatenate(cost_vanilla).mean()),
    }


def upstream_batches(split: str, batch_size: int):
    """Yield batches with starts sampled by the upstream dataset class.

    Args:
        split: Split name in upstream spelling (``"train"``, ``"valid"``,
            ``"test"``).
        batch_size: Batch size.

    Yields:
        tuple: ``(map_designs, start_maps, goal_maps)`` tensors.

    Note:
        Requires the global seed to be set beforehand; the start sampler uses
        the global `numpy.random` state (`MazeDataset.get_random_start_map`).
    """
    from neural_astar.utils.data import create_dataloader

    loader = create_dataloader(str(config.MAZE_NPZ), split, batch_size)
    for map_designs, start_maps, goal_maps, _ in loader:
        yield map_designs, start_maps, goal_maps


def fixed_batches(split: str, batch_size: int, seed: int):
    """Yield batches with our fixed, stored starts.

    Args:
        split: Split name (any spelling accepted by `src.config.canonical_split`).
        batch_size: Batch size.
        seed: Seed for `src.data.choose_start`.

    Yields:
        tuple: ``(map_designs, start_maps, goal_maps)`` tensors.
    """
    data = load_split(split)
    instances = data.instances(seed=seed)
    for begin in range(0, len(instances), batch_size):
        chunk = instances[begin : begin + batch_size]
        tensors = [inst.to_tensors() for inst in chunk]
        yield (
            torch.cat([t[0] for t in tensors]),
            torch.cat([t[1] for t in tensors]),
            torch.cat([t[2] for t in tensors]),
        )


def main() -> None:
    """Parse arguments, run both protocols, print and save the results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", help="train | val | test")
    parser.add_argument("--seed", type=int, default=0, help="global and start seed")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument(
        "--protocol",
        default="both",
        choices=["both", "upstream", "fixed"],
        help="start-selection protocol to evaluate",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=config.RESULTS_DIR / "e0_reproduce.json",
        help="where to write the JSON report",
    )
    args = parser.parse_args()

    split = config.canonical_split(args.split)
    device = config.get_device()
    config.set_global_seeds(args.seed)

    planner = load_pretrained_neural_astar(device=device)
    vanilla = make_vanilla_astar(device=device)

    report: Dict[str, object] = {
        "split": split,
        "seed": args.seed,
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "checkpoint": str(
            sorted(config.PRETRAINED_CKPT_DIR.glob("**/*.ckpt"))[-1].relative_to(
                config.REPO_ROOT
            )
        ),
        "protocols": {},
    }

    protocols = ["upstream", "fixed"] if args.protocol == "both" else [args.protocol]
    for protocol in protocols:
        config.set_global_seeds(args.seed)  # re-seed: upstream starts are global-RNG
        batches = (
            upstream_batches(split, args.batch_size)
            if protocol == "upstream"
            else fixed_batches(split, args.batch_size, args.seed)
        )
        started = time.perf_counter()
        metrics = evaluate_batches(planner, vanilla, batches)
        metrics["seconds"] = round(time.perf_counter() - started, 2)
        report["protocols"][protocol] = metrics  # type: ignore[index]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"\nE0 on split={split} (n={len(load_split(split))}), seed={args.seed}\n")
    header = f"| {'protocol':9} | {'Opt':>6} | {'Exp':>6} | {'Hmean':>6} | {'exp_na':>7} | {'exp_a*':>7} | {'s':>5} |"
    print(header)
    print("|" + "-" * (len(header) - 2) + "|")
    for name, m in report["protocols"].items():  # type: ignore[union-attr]
        print(
            f"| {name:9} | {m['p_opt']:6.3f} | {m['p_exp']:6.3f} | {m['h_mean']:6.3f} "
            f"| {m['mean_expansions_neural']:7.1f} | {m['mean_expansions_classical']:7.1f} "
            f"| {m['seconds']:5.1f} |"
        )
    print(f"\nwrote {args.out.relative_to(config.REPO_ROOT)}")


if __name__ == "__main__":
    main()
