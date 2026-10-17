"""PLAN.md step 4: build the E1 feature library for every instance.

For each split, computes the encoder's cost map and the per-cell features of
`src.features`, and saves them aligned in `results/features_{split}.npz`:
every array has shape ``(N, 32, 32)`` and index ``i`` refers to instance ``i``
of that split, with the fixed start recorded in `start_rc`.

Usage:
    python scripts/build_features.py --splits test --instances 3 --timing-only
    python scripts/build_features.py                  # all three splits
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.data import load_split  # noqa: E402
from src.features import AUX_NAMES, FEATURE_NAMES, compute_instance_features  # noqa: E402
from src.planner import encode_cost_map, load_pretrained_neural_astar  # noqa: E402


def build_split(
    split_name: str,
    seed: int,
    planner,
    limit: int | None = None,
    batch_size: int = 50,
    tag: str = "",
) -> Dict[str, np.ndarray]:
    """Compute cost maps and features for one split.

    Args:
        split_name: Split to process.
        seed: Start-selection seed (recorded in the archive).
        planner: Pretrained `NeuralAstar`.
        limit: Optional cap on the number of instances (for timing probes).
        batch_size: Instances per encoder forward pass.
        tag: Label for the model that produced the cost maps, recorded in the
            archive. Empty means the shipped pretrained checkpoint.

    Returns:
        dict[str, np.ndarray]: Arrays ready for `np.savez_compressed` —
        one ``(N, H, W)`` array per name in `FEATURE_NAMES` and `AUX_NAMES`,
        plus ``instance_ids``, ``start_rc``, ``goal_rc``, ``optimal_cost``,
        ``feature_names``, ``split`` and ``seed``.
    """
    split = load_split(split_name)
    instances = split.instances(seed=seed)
    if limit is not None:
        instances = instances[:limit]

    cost_maps: List[np.ndarray] = []
    for begin in range(0, len(instances), batch_size):
        chunk = instances[begin : begin + batch_size]
        tensors = [inst.to_tensors() for inst in chunk]
        maps = torch.cat([t[0] for t in tensors])
        starts = torch.cat([t[1] for t in tensors])
        goals = torch.cat([t[2] for t in tensors])
        with torch.no_grad():
            cost_maps.append(encode_cost_map(planner, maps, starts, goals)[:, 0].numpy())
    costs = np.concatenate(cost_maps) if cost_maps else np.zeros((0,))

    stacked: Dict[str, List[np.ndarray]] = {name: [] for name in FEATURE_NAMES + AUX_NAMES}
    for k, instance in enumerate(instances):
        reference = split.reference_trajectory(instance.instance_id, instance.start_rc)
        per_cell = compute_instance_features(
            instance, cost_map=costs[k], reference_path=reference
        )
        for name, array in per_cell.items():
            stacked[name].append(array)

    archive: Dict[str, np.ndarray] = {
        name: np.stack(arrays) for name, arrays in stacked.items() if arrays
    }
    archive["instance_ids"] = np.array([i.instance_id for i in instances], dtype=np.int32)
    archive["start_rc"] = np.array([i.start_rc for i in instances], dtype=np.int16)
    archive["goal_rc"] = np.array([i.goal_rc for i in instances], dtype=np.int16)
    archive["optimal_cost"] = np.array([i.optimal_cost for i in instances], dtype=np.float32)
    archive["feature_names"] = np.array(FEATURE_NAMES)
    archive["model_tag"] = np.array(tag)
    archive["split"] = np.array(split_name)
    archive["seed"] = np.array(seed)
    return archive


def main() -> int:
    """Build the requested splits, or just time a few instances."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits", nargs="+", default=["train", "valid", "test"], help="splits to build"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--instances", type=int, default=None, help="cap, for probes")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument(
        "--timing-only", action="store_true", help="time and extrapolate, write nothing"
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
        help="suffix for the archive filename, e.g. seed1 -> features_test_seed1.npz",
    )
    args = parser.parse_args()

    config.set_global_seeds(args.seed)
    planner = load_pretrained_neural_astar(checkpoint_path=args.checkpoint)

    for split_name in args.splits:
        canonical = config.canonical_split(split_name)
        started = time.perf_counter()
        archive = build_split(
            canonical,
            args.seed,
            planner,
            limit=args.instances,
            batch_size=args.batch_size,
            tag=args.tag,
        )
        seconds = time.perf_counter() - started
        n = len(archive["instance_ids"])
        per_instance = seconds / max(n, 1)
        print(
            f"{canonical}: {n} instances, {len(FEATURE_NAMES)} features + "
            f"{len(AUX_NAMES)} aux arrays in {seconds:.1f} s "
            f"({1000 * per_instance:.1f} ms per instance)"
        )
        if args.timing_only:
            full = config.SPLIT_SIZES[canonical]
            print(
                f"  extrapolation to all {full} instances: "
                f"{per_instance * full:.0f} s; all 1000: "
                f"{per_instance * sum(config.SPLIT_SIZES.values()):.0f} s"
            )
            continue

        suffix = f"_{args.tag}" if args.tag else ""
        out = config.RESULTS_DIR / f"features_{canonical}{suffix}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **archive)
        size_mb = out.stat().st_size / 1e6
        print(f"  wrote {out.relative_to(config.REPO_ROOT)} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
