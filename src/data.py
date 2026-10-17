"""Loading the maze benchmark with **fixed, stored** start cells.

Why this module exists at all: upstream's `MazeDataset.__getitem__` samples a
fresh random start every time it is called
(`neural_astar/utils/data.py:159-160`), so two runs of the same experiment see
different problems. The project requires the opposite — "for our experiments
**fix the start per instance** and store it, so results are reproducible" —
and every perturbation result in this project is keyed by
``(instance_id, start_rc)``.

Dataset facts used here (all verified in `docs/verified.md`):

* arrays per split are ``(map_designs, goal_maps, opt_policies, opt_dists)``
  at indices ``arr_{b}..arr_{b+3}``, ``b`` from `src.config.SPLIT_BASE_INDEX`;
* maps are ``(N, 32, 32)`` with ``1`` = free, ``0`` = obstacle;
* goal maps are ``(N, 1, 32, 32)`` one-hot;
* ``opt_dists`` are ``(N, 1, 32, 32)`` and stored **negated**, with the
  sentinel ``-1024`` for obstacles *and* for free cells outside the goal's
  connected component. This module returns sign-normalised distances with
  ``np.inf`` in place of the sentinel, so "unreachable" is never confusable
  with "very far".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from . import config
from .grid import RC

#: Default percentile band (of distance-to-goal, over reachable free cells)
#: that start cells are drawn from. Upstream samples uniformly from one of the
#: 55-70 / 70-85 / 85-100 percentile intervals
#: (`neural_astar/utils/data.py:200-217`), i.e. from the union [55, 100].
#: Keeping the same band keeps our fixed starts in-distribution with the
#: protocol the pretrained model was validated under.
START_PERCENTILE_BAND: Tuple[float, float] = (55.0, 100.0)


@dataclass(frozen=True)
class MazeInstance:
    """One planning problem: a map, a goal, a fixed start, and the reference.

    Attributes:
        instance_id: Index of the instance within its split.
        split: Canonical split name (``"train"``, ``"valid"``, ``"test"``).
        map_design: ``(H, W)`` float32 binary map; ``1`` = free.
        goal_rc: Goal cell ``(row, col)``.
        start_rc: Fixed start cell ``(row, col)``, chosen by
            `choose_start` and stored with every result.
        opt_dist: ``(H, W)`` float64 unit-cost distance-to-goal field from the
            dataset, sign-normalised, with ``np.inf`` where unreachable.
        start_seed: Seed that produced `start_rc`, recorded for the manifest.
    """

    instance_id: int
    split: str
    map_design: np.ndarray
    goal_rc: RC
    start_rc: RC
    opt_dist: np.ndarray
    start_seed: int

    @property
    def optimal_cost(self) -> float:
        """Unit-cost length of an optimal start->goal path on the *original* map.

        Read straight from the dataset's distance field, which makes it an
        independent reference for `src.exact_solver` (PLAN.md step 2 tests the
        solver against exactly this).

        Returns:
            float: Number of moves on a shortest path, or ``np.inf`` if the
            goal is unreachable from `start_rc`.
        """
        return float(self.opt_dist[self.start_rc])

    def to_tensors(self, device=None):
        """Return ``(map_designs, start_maps, goal_maps)`` shaped for the planner.

        The upstream planners expect ``(batch, 1, H, W)`` float tensors
        (`neural_astar/planner/differentiable_astar.py:172-175` asserts
        ``ndim == 4``).

        Args:
            device: Torch device; defaults to `src.config.get_device()`.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: One-instance
            batches of the map, the one-hot start map and the one-hot goal map.
        """
        import torch

        device = device or config.get_device()
        m = torch.from_numpy(np.ascontiguousarray(self.map_design, dtype=np.float32))
        start = torch.zeros_like(m)
        goal = torch.zeros_like(m)
        start[self.start_rc] = 1.0
        goal[self.goal_rc] = 1.0
        shape = (1, 1) + m.shape
        return (
            m.reshape(shape).to(device),
            start.reshape(shape).to(device),
            goal.reshape(shape).to(device),
        )


@dataclass(frozen=True)
class MazeSplit:
    """All instances of one split, as raw arrays plus convenience accessors.

    Attributes:
        split: Canonical split name.
        map_designs: ``(N, H, W)`` float32 binary maps.
        goal_maps: ``(N, 1, H, W)`` float32 one-hot goal maps.
        opt_policies: ``(N, 8, 1, H, W)`` float32 one-hot optimal directions
            (kept as-is; used only to reconstruct reference paths).
        opt_dists: ``(N, 1, H, W)`` float64 sign-normalised distances to goal,
            ``np.inf`` where unreachable.
        npz_path: File the arrays were read from.
    """

    split: str
    map_designs: np.ndarray
    goal_maps: np.ndarray
    opt_policies: np.ndarray
    opt_dists: np.ndarray
    npz_path: Path

    def __len__(self) -> int:
        """Number of instances in the split."""
        return int(self.map_designs.shape[0])

    def goal_rc(self, instance_id: int) -> RC:
        """Goal cell of one instance.

        Args:
            instance_id: Index within the split.

        Returns:
            tuple[int, int]: ``(row, col)`` of the single goal cell.

        Raises:
            ValueError: If the goal map is not one-hot.
        """
        hits = np.argwhere(self.goal_maps[instance_id, 0] == 1)
        if len(hits) != 1:
            raise ValueError(
                f"instance {instance_id}: expected exactly one goal cell, found {len(hits)}"
            )
        return (int(hits[0][0]), int(hits[0][1]))

    def instance(self, instance_id: int, seed: int = 0, rule: str = "band_rng") -> MazeInstance:
        """Build a `MazeInstance` with a deterministic start cell.

        Args:
            instance_id: Index within the split.
            seed: Seed for the start-cell rule. The same
                ``(split, instance_id, seed, rule)`` always yields the same
                start, across machines and runs.
            rule: Start-selection rule, see `choose_start`.

        Returns:
            MazeInstance: The problem instance, with the start stored.
        """
        dist = self.opt_dists[instance_id, 0]
        return MazeInstance(
            instance_id=instance_id,
            split=self.split,
            map_design=self.map_designs[instance_id],
            goal_rc=self.goal_rc(instance_id),
            start_rc=choose_start(dist, instance_id=instance_id, seed=seed, rule=rule),
            opt_dist=dist,
            start_seed=seed,
        )

    def reference_trajectory(self, instance_id: int, start_rc: RC) -> np.ndarray:
        """The dataset's optimal trajectory from `start_rc`, as a binary map.

        Args:
            instance_id: Index within the split.
            start_rc: Start cell (the trajectory depends on it).

        Returns:
            np.ndarray: Float32 ``(H, W)`` map; see `optimal_trajectory` for
            the convention (start marked, goal not).
        """
        return optimal_trajectory(
            self.opt_policies[instance_id], start_rc, self.goal_rc(instance_id)
        )

    def instances(self, seed: int = 0, rule: str = "band_rng") -> Sequence[MazeInstance]:
        """Build every instance of the split with fixed starts.

        Args:
            seed: Seed for the start-cell rule.
            rule: Start-selection rule, see `choose_start`.

        Returns:
            list[MazeInstance]: One entry per instance, in dataset order.
        """
        return [self.instance(i, seed=seed, rule=rule) for i in range(len(self))]


def load_split(split: str, npz_path: Optional[Path] = None) -> MazeSplit:
    """Load one split of the maze benchmark, normalising the distance sign.

    Args:
        split: ``"train"``, ``"val"``/``"valid"`` or ``"test"``.
        npz_path: Override for the npz file; defaults to
            `src.config.MAZE_NPZ`.

    Returns:
        MazeSplit: Arrays for the split. Distances are returned as
        non-negative move counts with ``np.inf`` for unreachable cells
        (the raw file stores ``-distance`` with a ``-1024`` sentinel).

    Raises:
        FileNotFoundError: If the npz file is missing (the data symlink points
            into the nested `planning-datasets` submodule; run
            ``git submodule update --init --recursive``).
    """
    path = Path(npz_path) if npz_path is not None else config.MAZE_NPZ
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Did you run `git submodule update --init --recursive`? "
            "See docs/setup.md."
        )
    canonical = config.canonical_split(split)
    keys = config.split_array_keys(canonical)
    with np.load(path) as f:
        map_designs = f[keys[0]].astype(np.float32)
        goal_maps = f[keys[1]].astype(np.float32)
        opt_policies = f[keys[2]].astype(np.float32)
        raw_dists = f[keys[3]].astype(np.float64)

    dists = normalize_opt_dists(raw_dists)
    return MazeSplit(
        split=canonical,
        map_designs=map_designs,
        goal_maps=goal_maps,
        opt_policies=opt_policies,
        opt_dists=dists,
        npz_path=path,
    )


def normalize_opt_dists(raw_dists: np.ndarray) -> np.ndarray:
    """Turn the dataset's negated distance arrays into usable distances.

    Args:
        raw_dists: Array as stored in the npz: non-positive values, where
            ``-1024`` (``-H*W``) marks obstacles and unreachable free cells.

    Returns:
        np.ndarray: Same shape, float64, with non-negative move counts and
        ``np.inf`` wherever the sentinel appeared.

    Note:
        The sentinel is compared against `src.config.UNREACHABLE_DIST` rather
        than "the array minimum", so a split whose maps happen to be fully
        connected is handled identically.
    """
    dists = -np.asarray(raw_dists, dtype=np.float64)
    dists[dists >= config.UNREACHABLE_DIST] = np.inf
    return dists


def choose_start(
    opt_dist: np.ndarray,
    instance_id: int,
    seed: int = 0,
    rule: str = "band_rng",
) -> RC:
    """Pick a reproducible start cell from the distance-to-goal field.

    Args:
        opt_dist: ``(H, W)`` sign-normalised distances to the goal
            (``np.inf`` = unreachable, ``0`` at the goal itself).
        instance_id: Index of the instance; mixed into the RNG so that
            different instances get independent draws from one `seed`.
        seed: Base seed.
        rule: One of

            * ``"band_rng"`` (default) — uniform draw, from a seeded
              `numpy.random.Generator`, among reachable cells whose distance
              lies in the `START_PERCENTILE_BAND` percentile band. This is the
              deterministic analogue of upstream's sampling scheme.
            * ``"farthest"`` — the reachable cell with the largest distance,
              ties broken by row-major order. Fully determined by the map (no
              seed), useful for worst-case/diagnostic runs.

    Returns:
        tuple[int, int]: The chosen start cell ``(row, col)``.

    Raises:
        ValueError: If no reachable cell other than the goal exists, or `rule`
            is unknown.
    """
    reachable = np.isfinite(opt_dist) & (opt_dist > 0)
    if not reachable.any():
        raise ValueError("instance has no reachable cell other than the goal")

    candidates = np.argwhere(reachable)
    values = opt_dist[reachable]

    if rule == "farthest":
        best = int(np.argmax(values))
        rc = candidates[best]
        return (int(rc[0]), int(rc[1]))

    if rule == "band_rng":
        lo, hi = np.percentile(values, START_PERCENTILE_BAND)
        in_band = (values >= lo) & (values <= hi)
        if not in_band.any():  # degenerate: all distances identical
            in_band = np.ones_like(values, dtype=bool)
        banded = candidates[in_band]
        rng = np.random.default_rng([seed, instance_id])
        rc = banded[int(rng.integers(len(banded)))]
        return (int(rc[0]), int(rc[1]))

    raise ValueError(f"unknown start rule {rule!r}; expected 'band_rng' or 'farthest'")


#: Upstream's action index -> ``(dr, dc)`` move, from
#: `neural_astar/utils/data.py:234-243` (`MazeDataset.next_loc`). The stored
#: policy arrays (`arr_2`/`arr_6`/`arr_10`) are one-hot over these eight
#: actions, so reproducing a reference trajectory requires this exact order.
ACTION_TO_MOVE: Tuple[Tuple[int, int], ...] = (
    (-1, 0),   # 0: up
    (0, +1),   # 1: right
    (0, -1),   # 2: left
    (+1, 0),   # 3: down
    (-1, +1),  # 4: up-right
    (-1, -1),  # 5: up-left
    (+1, +1),  # 6: down-right
    (+1, -1),  # 7: down-left
)


def optimal_trajectory(
    opt_policy: np.ndarray,
    start_rc: RC,
    goal_rc: RC,
    max_steps: Optional[int] = None,
) -> np.ndarray:
    """Reference trajectory from the dataset's optimal policy — the training target.

    This is the supervision signal Neural A\\* was trained on: the loss is
    ``L1(histories, opt_trajs)`` (`utils/training.py:63-66`), so the encoder was
    only ever asked to make the search *expand these cells*. That makes this
    indicator the sharpest available test of the competing hypothesis in
    PLAN.md — that the cost map is a path-prediction map rather than a
    heuristic.

    Replicates `MazeDataset.get_opt_traj` exactly, including the detail that
    the **goal cell is not marked** (the loop writes the current cell and then
    steps, stopping once the goal is reached) while the start cell is.

    Args:
        opt_policy: One-hot policy for one instance, shaped ``(8, 1, H, W)`` as
            stored in the npz, or ``(8, H, W)``.
        start_rc: Start cell.
        goal_rc: Goal cell.
        max_steps: Safety cap on the walk; defaults to ``H * W``.

    Returns:
        np.ndarray: Float32 ``(H, W)`` binary trajectory map.

    Raises:
        ValueError: If the policy revisits a cell (upstream asserts the same
            thing) or the goal is not reached within `max_steps`, either of
            which means the policy and the start are inconsistent.
    """
    policy = np.asarray(opt_policy)
    if policy.ndim == 4:
        policy = policy[:, 0]
    if policy.ndim != 3 or policy.shape[0] != len(ACTION_TO_MOVE):
        raise ValueError(f"expected a (8, 1, H, W) or (8, H, W) policy, got {policy.shape}")

    height, width = policy.shape[1:]
    traj = np.zeros((height, width), dtype=np.float32)
    cap = max_steps if max_steps is not None else height * width
    current = start_rc
    for _ in range(cap):
        if current == goal_rc:
            return traj
        if traj[current] != 0.0:
            raise ValueError(f"policy revisits {current}; start/policy mismatch")
        traj[current] = 1.0
        action = int(np.argmax(policy[:, current[0], current[1]]))
        move = ACTION_TO_MOVE[action]
        current = (current[0] + move[0], current[1] + move[1])
        if not (0 <= current[0] < height and 0 <= current[1] < width):
            raise ValueError(f"policy stepped outside the grid to {current}")
    raise ValueError(f"policy did not reach the goal within {cap} steps")
