"""Render the README figures: what a dataset sample looks like and how the search runs on it.

Writes two files under ``figures/`` that the README embeds:

* ``dataset_samples.png`` -- one row per map, four columns: (1) the map with
  the start (S), the goal (G) and the dataset's optimal path; (2) the cost map
  the encoder predicts; (3) the cells Neural A* expands and the path it
  returns; (4) the same for classical A*, the exact control.
* ``neural_astar_search.gif`` -- the search on the first map, one frame per
  expansion, with both planners advancing in lockstep by expansion count.

The maps are the first three test instances, with starts chosen by
`src.data.choose_start` at the given seed. They were not selected for looking
good; the printed summary reports the planners' results on them as they are.

Usage:
    python scripts/render_readme_figures.py --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.data import load_split  # noqa: E402
from src.exact_solver import solve_astar  # noqa: E402
from src.planner import (  # noqa: E402
    encode_cost_map,
    load_pretrained_neural_astar,
    make_vanilla_astar,
    run_astar_module,
)

FIG_DIR: Path = config.REPO_ROOT / "figures"

#: Test-split instances shown in the sample figure.
INSTANCE_IDS = (0, 1, 2)

#: Instance animated in the GIF.
GIF_INSTANCE = 0

#: At most this many frames; longer searches are subsampled.
MAX_GIF_FRAMES = 90
FRAME_MS = 70
HOLD_MS = 2000

# RGBA overlay colours (0-1 floats).
EXPANDED = (0.12, 0.47, 0.71, 0.35)
NEURAL_PATH = (1.0, 0.50, 0.05, 0.95)
CLASSICAL_PATH = (0.58, 0.40, 0.74, 0.95)
OPTIMAL_PATH = (0.0, 0.60, 0.60, 0.95)
CURRENT_NODE = (1.0, 0.90, 0.0, 1.0)


def to_np(tensor) -> np.ndarray:
    """Detach a planner tensor and return it as a numpy array."""
    return tensor.detach().cpu().numpy()


def run_both(planner, vanilla, instance) -> dict:
    """Run Neural A* and classical A* on one instance, keeping the search history.

    Args:
        planner: Pretrained `NeuralAstar`.
        vanilla: Classical `VanillaAstar` control.
        instance: A `MazeInstance` with its fixed start.

    Returns:
        dict: ``cost`` (the encoder's cost map), ``neural`` and ``classical``
        (`AstarOutput` with per-expansion history), and ``optimal`` (the exact
        unit-cost solution).
    """
    m, s, g = instance.to_tensors()
    cost = encode_cost_map(planner, m, s, g)
    neural = run_astar_module(planner, cost, s, g, m, store_intermediate_results=True)
    classical = run_astar_module(vanilla, m, s, g, m, store_intermediate_results=True)
    optimal = solve_astar(instance.map_design, instance.start_rc, instance.goal_rc)
    return {
        "cost": to_np(cost[0, 0]),
        "neural": neural,
        "classical": classical,
        "optimal": optimal,
    }


def path_moves(path_map: np.ndarray) -> int:
    """Moves on a returned path: one fewer than its cell count (8-connected, unit cost)."""
    return int(path_map.sum()) - 1


def overlay(ax, mask: np.ndarray, rgba) -> None:
    """Paint the cells where ``mask`` is 1 with one translucent colour."""
    layer = np.zeros(mask.shape + (4,))
    layer[mask > 0] = rgba
    ax.imshow(layer, interpolation="nearest")


def mark_endpoints(ax, instance) -> None:
    """Draw the start (S, green) and goal (G, red) cells."""
    (sr, sc), (gr, gc) = instance.start_rc, instance.goal_rc
    for (r, c), label, colour in (((sr, sc), "S", "#2ca02c"), ((gr, gc), "G", "#d62728")):
        ax.scatter([c], [r], s=90, c=colour, edgecolors="black", zorder=5)
        ax.text(c, r, label, color="white", ha="center", va="center", fontsize=7, zorder=6)


def clean_axes(ax) -> None:
    """Remove ticks: the grid is 32x32 cells and the coordinates are not the point."""
    ax.set_xticks([])
    ax.set_yticks([])


def draw_sample_row(axes, instance, run) -> None:
    """Fill one row of the sample figure (four panels for one map)."""
    free = instance.map_design
    neural_moves = path_moves(to_np(run["neural"].paths[0, 0]))
    classical_moves = path_moves(to_np(run["classical"].paths[0, 0]))
    optimal_moves = run["optimal"].n_moves
    expanded_neural = int(to_np(run["neural"].histories[0, 0]).sum())
    expanded_classical = int(to_np(run["classical"].histories[0, 0]).sum())

    ax = axes[0]
    ax.imshow(free, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    overlay(ax, _path_mask(run["optimal"].path, free.shape), OPTIMAL_PATH)
    mark_endpoints(ax, instance)
    ax.set_title(
        f"map {instance.instance_id}: S, G, optimal path ({optimal_moves} moves)\n"
        "white = free, black = obstacle",
        fontsize=9,
    )
    clean_axes(ax)

    ax = axes[1]
    im = ax.imshow(run["cost"], cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    mark_endpoints(ax, instance)
    ax.set_title("encoder cost map (dark = cheap)", fontsize=9)
    clean_axes(ax)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax, output, path_colour, label, n_exp, n_moves in (
        (axes[2], run["neural"], NEURAL_PATH, "Neural A*", expanded_neural, neural_moves),
        (axes[3], run["classical"], CLASSICAL_PATH, "classical A*", expanded_classical, classical_moves),
    ):
        ax.imshow(free, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        overlay(ax, to_np(output.histories[0, 0]), EXPANDED)
        overlay(ax, to_np(output.paths[0, 0]), path_colour)
        mark_endpoints(ax, instance)
        ax.set_title(
            f"{label}: {n_exp} expanded, {n_moves} moves\n"
            f"Opt = moves / optimal moves = {n_moves / optimal_moves:.3f}",
            fontsize=9,
        )
        clean_axes(ax)


def _path_mask(path, shape) -> np.ndarray:
    """Binary mask of an ordered list of ``(row, col)`` cells."""
    mask = np.zeros(shape, dtype=np.float32)
    for r, c in path:
        mask[r, c] = 1.0
    return mask


def render_samples(runs: dict, out_path: Path) -> None:
    """Write the four-panel sample figure, one row per instance."""
    fig, axes = plt.subplots(len(runs), 4, figsize=(13, 3.6 * len(runs)), dpi=110)
    for row, (instance_id, (instance, run)) in enumerate(runs.items()):
        draw_sample_row(axes[row], instance, run)
        axes[row][0].set_ylabel(f"test map {instance_id}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _gif_frame(fig, axes, instance, run, step: int) -> Image.Image:
    """Draw both panels at one lockstep expansion count and capture the canvas."""
    for ax in axes:
        ax.clear()
    free = instance.map_design
    nn_steps = run["neural"].intermediate_results
    cl_steps = run["classical"].intermediate_results

    nn_idx = min(step, len(nn_steps) - 1)
    cl_idx = min(step, len(cl_steps) - 1)
    nn_step, cl_step = nn_steps[nn_idx], cl_steps[cl_idx]

    ax = axes[0]
    ax.imshow(run["cost"], cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    overlay(ax, to_np(nn_step["histories"])[0, 0], EXPANDED)
    _draw_frontier(ax, nn_step, nn_idx == len(nn_steps) - 1)
    mark_endpoints(ax, instance)
    n_nn = int(to_np(nn_step["histories"])[0, 0].sum())
    ax.set_title(f"Neural A* on the cost map: {n_nn} expanded", fontsize=10)
    clean_axes(ax)

    ax = axes[1]
    ax.imshow(free, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    overlay(ax, to_np(cl_step["histories"])[0, 0], EXPANDED)
    _draw_frontier(ax, cl_step, cl_idx == len(cl_steps) - 1, colour=CLASSICAL_PATH)
    mark_endpoints(ax, instance)
    n_cl = int(to_np(cl_step["histories"])[0, 0].sum())
    ax.set_title(f"classical A* (exact control): {n_cl} expanded", fontsize=10)
    clean_axes(ax)

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return Image.fromarray(rgba[..., :3].copy())


def _draw_frontier(ax, step, is_final: bool, colour=NEURAL_PATH) -> None:
    """Show the node being expanded (yellow), or the finished path on the last frame."""
    overlay(ax, to_np(step["paths"])[0, 0], colour if is_final else CURRENT_NODE)


def render_gif(run: dict, instance, out_path: Path) -> int:
    """Write the search animation, one frame per lockstep expansion count.

    Returns:
        int: Number of frames written.
    """
    n_steps = max(len(run["neural"].intermediate_results), len(run["classical"].intermediate_results))
    steps = np.unique(np.linspace(0, n_steps - 1, min(n_steps, MAX_GIF_FRAMES)).round().astype(int))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.0), dpi=80)
    frames = [_gif_frame(fig, axes, instance, run, int(t)) for t in steps]
    plt.close(fig)
    durations = [FRAME_MS] * len(frames)
    durations[-1] = HOLD_MS
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    return len(frames)


def main() -> None:
    """Build the instances, run both planners, and write the two figures."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0, help="seed for starts and torch")
    parser.add_argument("--out-dir", type=Path, default=FIG_DIR)
    args = parser.parse_args()

    config.set_global_seeds(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    split = load_split("test")
    planner = load_pretrained_neural_astar()
    vanilla = make_vanilla_astar()

    instances = {i: split.instance(i, seed=args.seed) for i in INSTANCE_IDS}
    runs = {i: (inst, run_both(planner, vanilla, inst)) for i, inst in instances.items()}

    print(f"{'map':>4} {'opt moves':>9} {'NN moves':>8} {'Opt':>6} {'NN exp':>7} {'A* exp':>7}")
    for i, (inst, run) in runs.items():
        nn_moves = path_moves(to_np(run["neural"].paths[0, 0]))
        opt_moves = run["optimal"].n_moves
        print(
            f"{i:>4} {opt_moves:>9} {nn_moves:>8} {nn_moves / opt_moves:>6.3f} "
            f"{int(to_np(run['neural'].histories[0, 0]).sum()):>7} "
            f"{int(to_np(run['classical'].histories[0, 0]).sum()):>7}"
        )

    samples_path = args.out_dir / "dataset_samples.png"
    render_samples({i: runs[i] for i in INSTANCE_IDS}, samples_path)
    print(f"wrote {samples_path}")

    gif_inst, gif_run = runs[GIF_INSTANCE]
    gif_path = args.out_dir / "neural_astar_search.gif"
    n_frames = render_gif(gif_run, gif_inst, gif_path)
    print(f"wrote {gif_path} ({n_frames} frames)")


if __name__ == "__main__":
    main()
