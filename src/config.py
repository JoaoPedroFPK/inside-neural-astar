"""Single source of truth for paths, device, dataset layout and seeding.

Every script and test in this project imports its paths and its device from
here, so that no module hard-codes a filesystem layout or `"cuda"`
(project rule: "Everything runs on CPU by default. Use `torch.device` from a
single config; never hard-code `cuda`").

Facts about the dataset encoded in this module are verified in
`docs/verified.md` and regression-tested in `tests/test_data.py`.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

#: Repository root (this file lives in `<root>/src/config.py`).
REPO_ROOT: Path = Path(__file__).resolve().parents[1]

#: Upstream Neural A* checkout (git submodule; never edited, see the project's working rules).
SUBMODULE_DIR: Path = REPO_ROOT / "neural-astar"

#: Our data directory. `mazes_032_moore_c8.npz` is a symlink into the nested
#: `planning-datasets` submodule, so the bytes live in exactly one place.
DATA_DIR: Path = REPO_ROOT / "data"

#: The 32x32 maze benchmark used throughout E0-E3.
MAZE_NPZ: Path = DATA_DIR / "mazes_032_moore_c8.npz"

#: Directory holding the pretrained PyTorch-Lightning checkpoint shipped by
#: upstream for `mazes_032_moore_c8` (`load_pretrained_neural_astar` globs it).
PRETRAINED_CKPT_DIR: Path = (
    SUBMODULE_DIR / "model" / "mazes_032_moore_c8" / "lightning_logs"
)

#: Cached experiment artifacts (`.npz` / parquet). Later steps query these
#: instead of recomputing; see the working rules, "Save intermediate artifacts".
RESULTS_DIR: Path = REPO_ROOT / "results"

#: The results manifest written from PLAN.md step 6 onward.
MANIFEST_PARQUET: Path = RESULTS_DIR / "manifest.parquet"

DOCS_DIR: Path = REPO_ROOT / "docs"


# --------------------------------------------------------------------------- #
# Dataset layout
# --------------------------------------------------------------------------- #

#: Index of the first of the four arrays belonging to each split inside the
#: npz. Array `arr_{base}` is the map, `+1` the goal map, `+2` the optimal
#: policy and `+3` the optimal distances.
#:
#: Note the upstream spelling of the validation split is ``"valid"``
#: (`neural_astar.utils.data.MazeDataset._process`); we accept both ``"valid"``
#: and ``"val"`` in `src.data.load_split`.
SPLIT_BASE_INDEX: Dict[str, int] = {"train": 0, "valid": 4, "test": 8}

#: Number of instances per split in `mazes_032_moore_c8.npz`.
SPLIT_SIZES: Dict[str, int] = {"train": 800, "valid": 100, "test": 100}

#: Grid side length of the maze benchmark.
MAP_SIZE: int = 32

#: Value stored in the `opt_dists` arrays for cells that have no path to the
#: goal (obstacles, and free cells outside the goal's connected component).
#: Upstream `planning-datasets/planning_datasets_utils/dijkstra.py` initialises
#: distances to `H * W` and returns them negated, so the sentinel appears as
#: ``-1024`` in the raw array and as ``+1024`` once the sign is normalised.
UNREACHABLE_DIST: float = float(MAP_SIZE * MAP_SIZE)

#: Value of a *free* (traversable) cell in the binary maps. Obstacles are 0.
#: Verified in `docs/verified.md` (check 2) and relied upon by the A* module,
#: which multiplies candidate neighbours by the obstacle map as a mask.
FREE: float = 1.0
OBSTACLE: float = 0.0


# --------------------------------------------------------------------------- #
# Device and determinism
# --------------------------------------------------------------------------- #

def get_device(override: str | None = None):
    """Return the `torch.device` every module in this project should use.

    Resolution order: explicit `override` argument, then the
    ``INA_DEVICE`` environment variable, then ``"cpu"``. Planning in Neural A*
    is CPU-only work in this project; a GPU is only ever used for training
    (PLAN.md step 10), which runs on Colab/Kaggle rather than here.

    Args:
        override: Device string such as ``"cpu"``, ``"cuda"`` or ``"mps"``.
            Takes precedence over the environment variable.

    Returns:
        torch.device: The resolved device.

    Note:
        `torch` is imported lazily so that `src.config` stays importable in
        contexts that only need paths (e.g. plotting or dataset inspection).
    """
    import torch

    name = override or os.environ.get("INA_DEVICE", "cpu")
    return torch.device(name)


def set_global_seeds(seed: int) -> None:
    """Seed `random`, `numpy` and `torch` (CPU and CUDA) for reproducibility.

    Mirrors `neural_astar.utils.training.set_global_seeds` but without
    importing `pytorch_lightning` (see `src.planner` for why that matters), and
    additionally seeds `random`, which the upstream helper also does.

    Args:
        seed: Seed value. Every script in this project accepts ``--seed`` and
            calls this function before doing anything stochastic.
    """
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - no GPU in this project
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def canonical_split(split: str) -> str:
    """Normalise a split name to the upstream spelling.

    Args:
        split: ``"train"``, ``"val"``, ``"valid"`` or ``"test"``.

    Returns:
        str: ``"train"``, ``"valid"`` or ``"test"``.

    Raises:
        KeyError: If `split` is not a recognised split name.
    """
    alias = {"train": "train", "val": "valid", "valid": "valid", "test": "test"}
    if split not in alias:
        raise KeyError(f"unknown split {split!r}; expected one of {sorted(alias)}")
    return alias[split]


def split_array_keys(split: str) -> Tuple[str, str, str, str]:
    """Return the four npz keys (map, goal, policy, distances) for a split.

    Args:
        split: Split name, see `canonical_split`.

    Returns:
        tuple[str, str, str, str]: Keys ``(map_designs, goal_maps,
        opt_policies, opt_dists)`` into `MAZE_NPZ`.

    Example:
        >>> split_array_keys("test")
        ('arr_8', 'arr_9', 'arr_10', 'arr_11')
    """
    base = SPLIT_BASE_INDEX[canonical_split(split)]
    return tuple(f"arr_{base + offset}" for offset in range(4))  # type: ignore[return-value]
