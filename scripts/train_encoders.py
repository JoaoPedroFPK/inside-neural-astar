"""Train encoders — PLAN.md step 10 (seeds) and step 7b part (b) (families).

Wraps upstream's own training entry point rather than reimplementing it, so
the models this project analyses are produced exactly the way the published
ones were. Upstream's script is driven by hydra, so every setting is passed as
a ``key=value`` override.

Two rules enforced here:

* **The submodule is never written to.** Upstream defaults to saving
  checkpoints inside its own tree, which would leave the pinned submodule
  dirty and break the environment test. `logdir` is redirected to
  `results/models/` under this repository.
* **Training is timed before it is launched.** Pass ``--epochs 1`` first and
  read the extrapolation; a 50-epoch run on CPU is not something to start
  blind.

Usage:
    python scripts/train_encoders.py --seeds 0 --epochs 1          # timing probe
    python scripts/train_encoders.py --seeds 0 1 2 --epochs 50     # step 10
    python scripts/train_encoders.py --family forest --seeds 0     # step 7b(b)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402


def train_one(family: str, seed: int, epochs: int, logdir: Path, tmax: float) -> Dict[str, object]:
    """Run one upstream training job as a subprocess.

    Args:
        family: A key of `src.config.MAP_FAMILIES`.
        seed: Random seed for the run.
        epochs: Number of epochs.
        logdir: Where checkpoints go — outside the submodule.
        tmax: Search-budget fraction during training (upstream default 0.25).

    Returns:
        dict: Outcome with the command, the seconds taken, the return code and
        the checkpoint directory.

    Note:
        Upstream's hydra config sets ``chdir: False``, so the relative dataset
        path is resolved against the working directory; the job therefore runs
        with the submodule as its working directory.
    """
    dataset = f"planning-datasets/data/mpd/{config.MAP_FAMILIES[family]}"
    command = [
        str(config.REPO_ROOT / ".venv" / "bin" / "python"),
        "scripts/train.py",
        f"seed={seed}",
        f"dataset={dataset}",
        f"logdir={logdir}",
        f"params.num_epochs={epochs}",
        f"Tmax={tmax}",
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        command, cwd=config.SUBMODULE_DIR, capture_output=True, text=True
    )
    seconds = time.perf_counter() - started
    if completed.returncode != 0:
        print(completed.stdout[-2000:])
        print(completed.stderr[-2000:], file=sys.stderr)
    return {
        "family": family,
        "seed": seed,
        "epochs": epochs,
        "seconds": seconds,
        "returncode": completed.returncode,
        "checkpoint_dir": str(logdir / config.MAP_FAMILIES[family]),
        "tag": f"{family}_seed{seed}",
    }


def main() -> int:
    """Train the requested encoders and record timings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", default="mazes", choices=list(config.MAP_FAMILIES))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--tmax", type=float, default=0.25)
    parser.add_argument(
        "--logdir", type=Path, default=config.RESULTS_DIR / "models"
    )
    args = parser.parse_args()

    args.logdir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, object]] = []
    for seed in args.seeds:
        # One directory per (family, seed). Upstream appends the dataset name
        # and a run-numbered subdirectory, so without this every seed would
        # land in the same tree and the checkpoint loader - which globs and
        # takes the last match - would silently mix them.
        run_dir = (args.logdir / f"{args.family}_seed{seed}").resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"training {args.family} seed={seed} for {args.epochs} epoch(s)...", flush=True)
        outcome = train_one(args.family, seed, args.epochs, run_dir, args.tmax)
        results.append(outcome)
        status = "ok" if outcome["returncode"] == 0 else f"FAILED ({outcome['returncode']})"
        per_epoch = outcome["seconds"] / max(args.epochs, 1)
        print(
            f"  {status} in {outcome['seconds'] / 60:.1f} min "
            f"({per_epoch:.0f} s/epoch; 50 epochs would take "
            f"{per_epoch * 50 / 60:.0f} min)",
            flush=True,
        )

    record = config.RESULTS_DIR / "training_runs.json"
    history = json.loads(record.read_text()) if record.exists() else []
    history.extend(results)
    record.write_text(json.dumps(history, indent=2, default=float) + "\n")

    failures = [r for r in results if r["returncode"] != 0]
    print(f"\n{len(results) - len(failures)}/{len(results)} runs succeeded; "
          f"wrote {record.relative_to(config.REPO_ROOT)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
