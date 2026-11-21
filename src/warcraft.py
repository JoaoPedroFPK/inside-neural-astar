"""WarCraft 12x12 maps: loading, ground truth, planning and tile edits (PLAN.md step 11).

The WarCraft benchmark (Vlastelica et al., ICLR 2020, "Blackbox
Combinatorial Solvers") differs from the maze families in three ways that
matter here:

* **The input is an image**, ``96 x 96`` RGB, made of ``12 x 12`` terrain
  tiles of ``8 x 8`` pixels each. The encoder (upstream ``CNNDownSize``)
  reads the image and emits a ``12 x 12`` cost map, one value per tile.
* **There are no obstacles.** Every tile is traversable; what differs is its
  *true cost* (the dataset's ``vertex_weights``, e.g. grass cheap, water
  expensive). Upstream trains this model with ``learn_obstacles=True``, which
  hands the search an all-ones mask (`neural_astar/planner/astar.py:203-205`).
* **Start and goal are fixed** at the top-left and bottom-right tiles
  (`neural_astar/utils/data.py:284-292`).

Ground truth follows the benchmark's convention: the cost of a path is the sum
of the true weights of *every* tile it visits, start and goal included. Our
exact solver charges a node's cost when it is expanded as a parent, so it never
charges the goal; `optimal_cost` adds the goal's weight back. Because the goal
is the same tile on every path of an instance, this shifts every path cost of
that instance by the same constant and changes no ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from neural_astar.planner import NeuralAstar

from . import config
from .exact_solver import ExactSolution, solve_astar, solve_dijkstra
from .grid import RC
from .planner import encode_cost_map, load_ptl_state_dict, run_astar_module

#: Tiles per side, and pixels per tile side.
GRID: int = 12
TILE_PX: int = 8

#: Fixed endpoints, as upstream's `WarCraftDataset.__getitem__` sets them.
START_RC: RC = (0, 0)
GOAL_RC: RC = (GRID - 1, GRID - 1)

#: Upstream file prefixes per split. The validation split is ``val`` on disk.
SPLIT_FILES = {"train": "train", "valid": "val", "val": "val", "test": "test"}

#: The configuration upstream trains the WarCraft model with
#: (`neural-astar/scripts/config/train_warcraft.yaml`, `train_warcraft.py`).
ENCODER_KW = dict(
    encoder_input="rgb+", encoder_arch="CNNDownSize", encoder_depth=3, const=10.0
)


@dataclass(frozen=True)
class WarcraftSplit:
    """One split of the 12x12 WarCraft data.

    Attributes:
        split: Canonical split name.
        images: ``(N, 96, 96, 3)`` uint8 terrain images.
        weights: ``(N, 12, 12)`` float64 true per-tile cost, one of
            0.8, 1.2, 5.3, 7.7 or 9.2.
        paths: ``(N, 12, 12)`` binary reference shortest path from the dataset.
    """

    split: str
    images: np.ndarray
    weights: np.ndarray
    paths: np.ndarray

    def __len__(self) -> int:
        return int(self.images.shape[0])


def load_warcraft_split(split: str, directory: Optional[Path] = None) -> WarcraftSplit:
    """Load one split from the extracted ``12x12`` directory.

    Args:
        split: ``train``, ``valid``/``val`` or ``test``.
        directory: Defaults to `src.config.WARCRAFT_DIR`.

    Returns:
        WarcraftSplit: Images, true weights and reference paths.
    """
    directory = Path(directory or config.WARCRAFT_DIR)
    prefix = SPLIT_FILES[split]
    return WarcraftSplit(
        split="valid" if prefix == "val" else split,
        images=np.load(directory / f"{prefix}_maps.npy"),
        # Stored as float16 (0.7998 for 0.8, ...). The reference paths were
        # computed with the exact one-decimal values, and only those make them
        # optimal on every map; the float16 values break 758 of 12 000 ties.
        weights=np.round(
            np.load(directory / f"{prefix}_vertex_weights.npy").astype(np.float64), 1
        ),
        paths=np.load(directory / f"{prefix}_shortest_paths.npy").astype(np.float32),
    )


# --------------------------------------------------------------------------- #
# Ground truth
# --------------------------------------------------------------------------- #

_FREE = np.ones((GRID, GRID), dtype=np.float32)


def optimal_solution(weights: np.ndarray) -> ExactSolution:
    """Exact cheapest start->goal path under the true weights (Dijkstra)."""
    return solve_dijkstra(_FREE, START_RC, GOAL_RC, cost_map=weights)


def optimal_cost(weights: np.ndarray) -> float:
    """True cost of a cheapest path: sum of weights over every visited tile."""
    return optimal_solution(weights).cost + float(weights[GOAL_RC])


def classical_expansions(weights: np.ndarray) -> int:
    """Expansions of classical A* given the *true* weights as its cost map.

    The same rule as the planner's search (upstream heuristic, ``g_ratio``
    0.5, cost added to ``h``), but fed the ground-truth costs instead of
    learned ones: the WarCraft analogue of the maze control, which feeds the
    search the true (unit) costs.
    """
    return solve_astar(_FREE, START_RC, GOAL_RC, cost_map=weights).expansions


def path_map_cost(path_map: np.ndarray, weights: np.ndarray) -> float:
    """True cost of a path given as a binary ``(12, 12)`` map."""
    return float((np.asarray(path_map) > 0.5).astype(np.float64).ravel() @ weights.ravel())


def is_connected_path(path_map: np.ndarray) -> bool:
    """Whether the marked cells contain an 8-connected start->goal chain."""
    marked = np.asarray(path_map) > 0.5
    if not (marked[START_RC] and marked[GOAL_RC]):
        return False
    seen = {START_RC}
    frontier = [START_RC]
    while frontier:
        r, c = frontier.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < GRID and 0 <= nc < GRID and marked[nr, nc] and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    frontier.append((nr, nc))
    return GOAL_RC in seen


# --------------------------------------------------------------------------- #
# Planner
# --------------------------------------------------------------------------- #

def load_warcraft_planner(checkpoint_path: Path, device=None) -> NeuralAstar:
    """Instantiate Neural A* with the WarCraft architecture and load weights.

    Args:
        checkpoint_path: A ``.ckpt`` file or a directory searched for one.
        device: Torch device; defaults to `src.config.get_device()`.

    Returns:
        NeuralAstar: In ``eval()`` mode (full search budget).
    """
    device = device or config.get_device()
    planner = NeuralAstar(learn_obstacles=True, **ENCODER_KW).to(device)
    planner.load_state_dict(load_ptl_state_dict(checkpoint_path))
    planner.eval()
    return planner


def to_tensors(images: np.ndarray, device=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Images ``(B, 96, 96, 3)`` -> planner inputs, exactly as upstream builds them.

    Accepts uint8 (scaled by 1/255, as upstream) or float already in [0, 1].
    """
    device = device or config.get_device()
    x = np.asarray(images)
    if x.ndim == 3:
        x = x[None]
    x = x.astype(np.float32) / 255.0 if x.dtype == np.uint8 else x.astype(np.float32)
    maps = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2))).to(device)
    starts = torch.zeros(x.shape[0], 1, GRID, GRID, device=device)
    goals = torch.zeros_like(starts)
    starts[:, 0, START_RC[0], START_RC[1]] = 1.0
    goals[:, 0, GOAL_RC[0], GOAL_RC[1]] = 1.0
    return maps, starts, goals


@torch.no_grad()
def plan(planner: NeuralAstar, images: np.ndarray, batch_size: int = 256):
    """Encode and search a batch of images.

    Returns:
        tuple: ``(cost_maps, paths, histories)``, each ``(B, 12, 12)`` numpy.
        The search gets an all-ones mask, as upstream does with
        ``learn_obstacles=True``.
    """
    costs, paths, hists = [], [], []
    for begin in range(0, len(images), batch_size):
        maps, starts, goals = to_tensors(images[begin : begin + batch_size])
        cost_maps = encode_cost_map(planner, maps, starts, goals)
        out = run_astar_module(planner, cost_maps, starts, goals, torch.ones_like(starts))
        costs.append(cost_maps[:, 0].cpu().numpy())
        paths.append(out.paths[:, 0].cpu().numpy())
        hists.append(out.histories[:, 0].cpu().numpy())
    return np.concatenate(costs), np.concatenate(paths), np.concatenate(hists)


@torch.no_grad()
def search_cost_maps(planner: NeuralAstar, cost_maps: np.ndarray):
    """Run only the search module on given ``(B, 12, 12)`` cost maps.

    Returns:
        tuple: ``(paths, histories)`` as ``(B, 12, 12)`` numpy.
    """
    device = config.get_device()
    c = torch.from_numpy(np.ascontiguousarray(cost_maps, dtype=np.float32))[:, None].to(device)
    starts = torch.zeros_like(c)
    goals = torch.zeros_like(c)
    starts[:, 0, START_RC[0], START_RC[1]] = 1.0
    goals[:, 0, GOAL_RC[0], GOAL_RC[1]] = 1.0
    out = run_astar_module(planner, c, starts, goals, torch.ones_like(starts))
    return out.paths[:, 0].cpu().numpy(), out.histories[:, 0].cpu().numpy()


# --------------------------------------------------------------------------- #
# Tile edits
# --------------------------------------------------------------------------- #

def tile_slice(rc: RC) -> Tuple[slice, slice]:
    """Pixel rows/cols of tile ``rc`` in a ``96 x 96`` image."""
    r, c = rc
    return slice(r * TILE_PX, (r + 1) * TILE_PX), slice(c * TILE_PX, (c + 1) * TILE_PX)


def transplant_tiles(
    image: np.ndarray,
    weights: np.ndarray,
    cells: Sequence[RC],
    donor_images: Sequence[np.ndarray],
    donor_cells: Sequence[RC],
    donor_weights: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Replace tiles with tiles cut from other maps, updating the true weights.

    The discrete analogue of flipping a maze cell: the terrain of each tile in
    `cells` becomes the terrain of a donor tile, and so does its true cost.

    Returns:
        tuple: ``(altered_image, altered_weights)``; inputs are not modified.
    """
    out_img = np.array(image, copy=True)
    out_w = np.array(weights, copy=True)
    for rc, donor, drc, dw in zip(cells, donor_images, donor_cells, donor_weights):
        out_img[tile_slice(rc)] = donor[tile_slice(drc)]
        out_w[rc] = dw
    return out_img, out_w


# --------------------------------------------------------------------------- #
# Feature library (the step-4 analogue)
# --------------------------------------------------------------------------- #

#: The five terrain costs of the benchmark, in increasing order.
TERRAIN_COSTS: Tuple[float, ...] = (0.8, 1.2, 5.3, 7.7, 9.2)

#: Feature groups for the surrogate fits. ``geometric`` is what the search's
#: own heuristic sees (position only); ``terrain`` is the true per-tile cost a
#: perfect cost estimator would output; ``path`` is what a path predictor
#: would output.
FEATURE_GROUPS = {
    "geometric": ("cheb_to_goal", "cheb_to_start"),
    "terrain": ("is_0.8", "is_1.2", "is_5.3", "is_7.7", "is_9.2", "weight_mean3"),
    "path": ("on_optimal_path", "cheb_to_optimal_path", "log_excess"),
}
FEATURE_GROUPS["terrain+geometric"] = FEATURE_GROUPS["terrain"] + FEATURE_GROUPS["geometric"]
FEATURE_GROUPS["terrain+path"] = FEATURE_GROUPS["terrain"] + FEATURE_GROUPS["path"]
FEATURE_GROUPS["all"] = (
    FEATURE_GROUPS["terrain"] + FEATURE_GROUPS["geometric"] + FEATURE_GROUPS["path"]
)


def through_costs(weights: np.ndarray) -> np.ndarray:
    """True cost of the cheapest start->goal path forced through each tile.

    ``D_s(v) + D_g(v) - w(v)``, where ``D_s``/``D_g`` are cheapest costs from
    the start/goal to ``v`` counting every visited tile (both ends included);
    ``w(v)`` is subtracted once because both halves count it. The minimum over
    all tiles is the optimum, attained exactly on the tiles of optimal paths.
    """
    from .exact_solver import distance_field

    to_goal = distance_field(_FREE, GOAL_RC, cost_map=weights) + weights[GOAL_RC]
    to_start = distance_field(_FREE, START_RC, cost_map=weights) + weights[START_RC]
    return to_start + to_goal - weights


def tile_features(weights: np.ndarray) -> dict:
    """Per-tile interpretable features for one map, each a ``(12, 12)`` array.

    Features:
        ``is_<c>``: one-hot terrain class.
        ``weight_mean3``: mean true cost over the 3x3 neighbourhood (edges
            replicated), the context an 8-pixel-wide receptive field sees.
        ``cheb_to_goal`` / ``cheb_to_start``: Chebyshev distance in tiles,
            the quantity the search heuristic is built from.
        ``on_optimal_path``: 1 on every tile of *some* optimal path.
        ``cheb_to_optimal_path``: Chebyshev distance to the nearest such tile.
        ``log_excess``: ``log(1 + through - optimum)``, how much more a path
            forced through the tile costs than the optimum.
    """
    from scipy.ndimage import uniform_filter

    rows, cols = np.indices((GRID, GRID))
    through = through_costs(weights)
    excess = np.maximum(through - through.min(), 0.0)
    on_path = excess < 1e-6  # float sums of one-decimal weights
    pr, pc = np.nonzero(on_path)
    cheb_to_path = np.min(
        np.maximum(np.abs(rows[..., None] - pr), np.abs(cols[..., None] - pc)), axis=-1
    )
    features = {f"is_{c}": np.isclose(weights, c).astype(np.float64) for c in TERRAIN_COSTS}
    features.update(
        weight_mean3=uniform_filter(weights, size=3, mode="nearest"),
        cheb_to_goal=np.maximum(np.abs(rows - GOAL_RC[0]), np.abs(cols - GOAL_RC[1])).astype(float),
        cheb_to_start=np.maximum(np.abs(rows - START_RC[0]), np.abs(cols - START_RC[1])).astype(float),
        on_optimal_path=on_path.astype(np.float64),
        cheb_to_optimal_path=cheb_to_path.astype(np.float64),
        log_excess=np.log1p(excess),
    )
    return features


def design_matrix(weights: np.ndarray, columns: Sequence[str]) -> np.ndarray:
    """Stack features of many maps into ``(N * 144, len(columns))`` rows."""
    blocks = []
    for w in weights:
        f = tile_features(w)
        blocks.append(np.stack([f[c].ravel() for c in columns], axis=1))
    return np.concatenate(blocks)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def exact_path_on(cost_map: np.ndarray) -> np.ndarray:
    """Binary map of the cheapest path under a *learned* cost map (Dijkstra).

    Used to separate the two ways the planner can go wrong: if this exact
    search on the encoder's costs already picks a worse path, the costs
    themselves rank paths wrongly; if it picks an optimal one, the deviation
    came from the planner's own search rule (heuristic and ``g_ratio``).
    """
    sol = solve_dijkstra(_FREE, START_RC, GOAL_RC, cost_map=np.asarray(cost_map, np.float64))
    out = np.zeros((GRID, GRID), dtype=np.float32)
    for rc in sol.path:
        out[rc] = 1.0
    return out


def score(
    weights: np.ndarray,
    paths: np.ndarray,
    histories: np.ndarray,
    cost_maps: np.ndarray,
) -> dict:
    """Project metrics for a batch of planner outputs on their true weights.

    Args:
        weights: ``(B, 12, 12)`` true costs each path is scored against.
        paths, histories, cost_maps: ``(B, 12, 12)`` planner outputs.

    Returns:
        dict of ``(B,)`` arrays: ``opt`` (true path cost / optimum), ``exp``
        (expansions / classical A* expansions on the true weights),
        ``opt_exact`` (Opt of an exact search on the learned costs),
        ``valid`` (connected start->goal path), plus the raw ``path_cost``,
        ``optimum``, ``expansions`` and ``classical_expansions``.
    """
    n = len(weights)
    out = {k: np.empty(n) for k in (
        "opt", "exp", "opt_exact", "path_cost", "optimum", "expansions", "classical_expansions"
    )}
    out["valid"] = np.empty(n, dtype=bool)
    for i in range(n):
        w = weights[i]
        optimum = optimal_cost(w)
        classical = classical_expansions(w)
        cost = path_map_cost(paths[i], w)
        out["path_cost"][i] = cost
        out["optimum"][i] = optimum
        out["expansions"][i] = float(histories[i].sum())
        out["classical_expansions"][i] = classical
        out["opt"][i] = cost / optimum
        out["exp"][i] = out["expansions"][i] / classical
        out["opt_exact"][i] = path_map_cost(exact_path_on(cost_maps[i]), w) / optimum
        out["valid"][i] = is_connected_path(paths[i])
    return out


#: Tolerance for "the path is optimal": one-decimal weights summed in float64.
OPT_TOL: float = 1e-6


# --------------------------------------------------------------------------- #
# Tile-transplant alterations (the step-3/6 analogue)
# --------------------------------------------------------------------------- #

#: Radius, in tiles, beyond which a tile edit cannot change the cost map.
#: ``CNNDownSize`` at depth 3 has a 38-pixel receptive field per output tile
#: (four 3x3 convolutions interleaved with three 2x2 max-pools), which reaches
#: at most two tiles in each direction. Checked on every alteration.
RECEPTIVE_RADIUS: int = 2

#: Every tile except the two endpoints may be edited.
EDITABLE_TILES: Tuple[RC, ...] = tuple(
    (r, c) for r in range(GRID) for c in range(GRID) if (r, c) not in (START_RC, GOAL_RC)
)


@dataclass(frozen=True)
class Alteration:
    """Tiles to replace and where their replacement textures come from.

    Attributes:
        cells: Tiles of the target map to replace.
        new_weights: True cost of each replacement (the donor tile's class).
        donors: ``(donor_map_index, row, col)`` per cell, indexing the donor
            split (the train split, so test textures are never reused).
    """

    cells: Tuple[RC, ...]
    new_weights: Tuple[float, ...]
    donors: Tuple[Tuple[int, int, int], ...]


def donor_pool(donor_weights: np.ndarray) -> dict:
    """Index every tile of the donor split by its terrain class.

    Returns:
        dict: terrain cost -> ``(M, 3)`` int array of ``(map, row, col)``.
    """
    pool = {}
    for c in TERRAIN_COSTS:
        pool[c] = np.argwhere(np.isclose(donor_weights, c))
    return pool


def draw_donor(pool: dict, cls: float, rng: np.random.Generator) -> Tuple[int, int, int]:
    """A random donor tile of terrain class `cls`."""
    choice = pool[cls][rng.integers(len(pool[cls]))]
    return int(choice[0]), int(choice[1]), int(choice[2])


def k1_alterations(weights: np.ndarray, pool: dict, rng: np.random.Generator) -> list:
    """Every single-tile edit of a map: each editable tile, each other class."""
    out = []
    for rc in EDITABLE_TILES:
        for cls in TERRAIN_COSTS:
            if np.isclose(weights[rc], cls):
                continue
            out.append(Alteration((rc,), (cls,), (draw_donor(pool, cls, rng),)))
    return out


def random_alterations(
    weights: np.ndarray, k: int, pool: dict, rng: np.random.Generator
) -> Alteration:
    """`k` distinct editable tiles, each given a random *different* class."""
    idx = rng.choice(len(EDITABLE_TILES), size=k, replace=False)
    cells = tuple(EDITABLE_TILES[i] for i in idx)
    classes = []
    for rc in cells:
        options = [c for c in TERRAIN_COSTS if not np.isclose(weights[rc], c)]
        classes.append(float(options[rng.integers(len(options))]))
    donors = tuple(draw_donor(pool, c, rng) for c in classes)
    return Alteration(cells, tuple(classes), donors)


def apply_alteration(
    image: np.ndarray, weights: np.ndarray, alteration: Alteration, donor_images: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Altered image and altered true weights."""
    return transplant_tiles(
        image,
        weights,
        alteration.cells,
        [donor_images[d[0]] for d in alteration.donors],
        [(d[1], d[2]) for d in alteration.donors],
        alteration.new_weights,
    )


def evaluate_alterations(
    planner: NeuralAstar,
    image: np.ndarray,
    weights: np.ndarray,
    alterations: Sequence[Alteration],
    donor_images: np.ndarray,
) -> Tuple[dict, list]:
    """Score one map's alterations against the altered maps' own optimum.

    Returns:
        tuple: ``(baseline, rows)``. ``baseline`` holds the unaltered map's
        scores (``C0``) and its cost map and path. Each row carries, for one
        alteration: ``opt_C1``/``exp_C1`` (the planner on the altered image,
        scored on the altered weights), ``opt_exact_C1`` (exact search on the
        altered cost map), ``opt_C0_on_W1`` (the *unaltered* planner path
        scored on the altered weights: the cost of not reacting at all), the
        altered optimum and classical expansions, and the cost-map change
        (``dcost_l1``, its share on the edited tiles, and its largest value
        beyond `RECEPTIVE_RADIUS`, which must be zero).
    """
    altered = [apply_alteration(image, weights, a, donor_images) for a in alterations]
    # The unaltered image rides in the same batch: convolution kernels chosen
    # per batch size differ by ~1e-4, which would otherwise show up as a
    # cost-map change on tiles the edit cannot reach.
    images = np.stack([image] + [a[0] for a in altered])
    costs, paths, hists = plan(planner, images, batch_size=len(images))
    c0, p0, h0 = costs[:1], paths[:1], hists[:1]
    c1, p1, h1 = costs[1:], paths[1:], hists[1:]
    base = score(weights[None], p0, h0, c0)
    baseline = {k: float(v[0]) for k, v in base.items()}
    baseline.update(cost_map=c0[0], path=p0[0])

    new_w = np.stack([a[1] for a in altered])
    s1 = score(new_w, p1, h1, c1)

    rows = []
    rr, cc = np.indices((GRID, GRID))
    for i, alt in enumerate(alterations):
        dcost = np.abs(c1[i] - c0[0])
        total = float(dcost.sum())
        near = np.zeros((GRID, GRID), dtype=bool)
        under = 0.0
        for rc in alt.cells:
            near |= np.maximum(np.abs(rr - rc[0]), np.abs(cc - rc[1])) <= RECEPTIVE_RADIUS
            under += float(dcost[rc])
        rows.append(
            {
                "flip_cells": [tuple(map(int, rc)) for rc in alt.cells],
                "budget": len(alt.cells),
                "original_weight": [float(weights[rc]) for rc in alt.cells],
                "new_weight": list(alt.new_weights),
                "donors": [list(d) for d in alt.donors],
                "opt_C1": float(s1["opt"][i]),
                "exp_C1": float(s1["exp"][i]),
                "opt_exact_C1": float(s1["opt_exact"][i]),
                "opt_C0_on_W1": path_map_cost(p0[0], new_w[i]) / float(s1["optimum"][i]),
                "path_valid_C1": bool(s1["valid"][i]),
                "path_changed": bool(np.any(p1[i] != p0[0])),
                "classical_opt_cost_altered": float(s1["optimum"][i]),
                "classical_expansions_altered": float(s1["classical_expansions"][i]),
                "expansions_C1": float(s1["expansions"][i]),
                "dcost_l1": total,
                "dcost_frac_under_flips": under / total if total > 0 else np.nan,
                "dcost_outside_receptive_field": float(dcost[~near].max(initial=0.0)),
            }
        )
    return baseline, rows
