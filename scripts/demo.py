"""Interactive demo: flip cells of a maze and watch Neural A* respond.

Click any cell to flip it (free <-> obstacle) and every panel updates from the
real pretrained planner:

* **map** — obstacles, the cells Neural A* expanded, its route, and the true
  optimal route on the map as it now stands (classical exact search);
* **cost map** — what the network outputs for every cell, and how much each
  cell's cost moved because of the flips (only cells within the network's
  11 x 11 receptive field of a flip can move);
* **verdict** — the project's error rule: the planner fails only if its route
  is longer than the *altered* map's optimum; finding a new route because the
  map changed is the correct response;
* **where the damage comes from** — the search rerun with only the network's
  costs changed and with only the obstacles changed (the 2x2 swap of step 7).

Buttons (and keys): worst single cell for path / effort, looked up in the
exhaustive sweep (`w` / `e`); one round of the gradient-guided attack of step
13 (`g` / `h`); a random flip (`r`); undo (`u`); reset (`0`); previous / next
map (`[` / `]`).

Usage:
    python scripts/demo.py                       # interactive window
    python scripts/demo.py --map 7
    python scripts/demo.py --actions "worst:opt grad:opt" --snapshots out/   # headless
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.data import load_split  # noqa: E402
from src.exact_solver import solve_astar, solve_dijkstra  # noqa: E402
from src.grid import RC, flip_cells, is_reachable  # noqa: E402
from src.perturb import (  # noqa: E402
    OPT_CAP_FOR_EFFICIENCY,
    candidate_cells,
    gradient_pool,
    greedy_flip_search,
)
from src.planner import encode_cost_map, load_pretrained_neural_astar, run_astar_module  # noqa: E402
from src.swap import run_swap  # noqa: E402

TOL = 1e-6
RECEPTIVE_RADIUS = 5
DETECTOR_CACHE = config.RESULTS_DIR / "cache" / "demo_detector.pkl"


class TamperDetector:
    """Step 12's detector, fitted once and cached, for the demo's tamper light.

    Exactly step 12's setup: a convolutional network that reads only the map
    (trained on random single flips of the 800 training maps), plus seven
    statistics of the cost map's residual from a geometry-only surrogate,
    combined by a logistic regression fitted on test maps 0-49 (clean vs.
    their worst search-effort flip). Maps 50-99 were never seen by it.
    """

    TRAIN_MAPS = 50

    def __init__(self, planner, seed: int = 0) -> None:
        import pickle

        self.planner = planner
        if DETECTOR_CACHE.exists():
            parts = pickle.loads(DETECTOR_CACHE.read_bytes())
        else:
            parts = self._fit(seed)
            DETECTOR_CACHE.parent.mkdir(parents=True, exist_ok=True)
            DETECTOR_CACHE.write_bytes(pickle.dumps(parts))
        self.__dict__.update(parts)
        from scripts.e4_detector import make_feature_builder

        self.builder = make_feature_builder("geometric_h1")

    def _fit(self, seed: int) -> Dict[str, object]:
        import pandas as pd

        from scripts.e4_detector import (
            clean_residual_sd, make_feature_builder, select_alterations, train_grid_control,
        )
        from src.detect import fit_detector, grid_cnn_inputs, residual_features, score_grid_cnn
        from src.surrogate import build_design_matrix, fit_and_score

        print("fitting the tamper detector once (about two minutes); cached afterwards", flush=True)
        surrogate = fit_and_score(build_design_matrix("train"), build_design_matrix("test"),
                                  "geometric_h1", kind="gbm", seed=seed)
        builder = make_feature_builder("geometric_h1")
        spread = clean_residual_sd(self.planner, surrogate, load_split("valid").instances(seed=seed), builder)
        cnn, _ = train_grid_control("train", "mazes", seed, 40)
        manifest = pd.read_parquet(config.RESULTS_DIR / "manifest.parquet")
        worst = select_alterations(manifest[manifest["split"] == "test"], "k1_eff")
        instances = load_split("test").instances(seed=seed)
        rows, labels = [], []
        for inst in instances[: self.TRAIN_MAPS]:
            if inst.instance_id not in worst:
                continue
            for label, design in ((0, inst.map_design), (1, flip_cells(inst.map_design, worst[inst.instance_id]))):
                column = score_grid_cnn(cnn, grid_cnn_inputs([inst], [design]))[0]
                rows.append([column] + residual_features(self.planner, surrogate, inst, design, builder, spread))
                labels.append(label)
        x, y = np.array(rows), np.array(labels)
        return {"surrogate": surrogate, "spread": spread, "cnn": cnn,
                "combined": fit_detector(x, y), "control": fit_detector(x[:, :1], y)}

    def score(self, instance, design: np.ndarray) -> Tuple[float, float]:
        """P(altered) from the map-only control and from control + cost-map residual."""
        from src.detect import grid_cnn_inputs, residual_features, score_grid_cnn

        column = score_grid_cnn(self.cnn, grid_cnn_inputs([instance], [design]))[0]
        x = np.array([[column] + residual_features(self.planner, self.surrogate, instance, design,
                                                   self.builder, self.spread)])
        return float(self.control.predict_proba(x[:, :1])[0, 1]), float(self.combined.predict_proba(x)[0, 1])


def order_path(path_map: np.ndarray, start: RC) -> List[RC]:
    """Order the cells of a binary path map from the start, for drawing a line.

    The search returns its route as a set of marked cells. A route is a chain
    of 8-connected cells, so it is walked from the start, always to the one
    unvisited marked neighbour.

    Args:
        path_map: ``(H, W)`` binary map of the route.
        start: The start cell, which lies on the route.

    Returns:
        list[tuple[int, int]]: The route in order; stops early if the marked
        cells are not a simple chain.
    """
    cells = {tuple(rc) for rc in np.argwhere(path_map > 0)}
    order = [tuple(start)]
    cells.discard(tuple(start))
    while cells:
        r, c = order[-1]
        nxt = [rc for rc in cells if max(abs(rc[0] - r), abs(rc[1] - c)) == 1]
        if not nxt:
            break
        # Prefer orthogonal steps when two neighbours qualify, as a route would.
        nxt.sort(key=lambda rc: abs(rc[0] - r) + abs(rc[1] - c))
        order.append(nxt[0])
        cells.discard(nxt[0])
    return order


class DemoState:
    """Everything the demo shows for one map and one set of flipped cells.

    Kept separate from the drawing so it can be tested and scripted.
    """

    def __init__(self, planner, split: str = "test", seed: int = 0,
                 manifest_path: Optional[Path] = None, detector: Optional[TamperDetector] = None) -> None:
        self.planner = planner
        self.detector = detector
        self.split = load_split(split)
        self.seed = seed
        self.manifest_path = manifest_path or config.RESULTS_DIR / "manifest.parquet"
        self._manifest = None
        self.instance_id = 0
        self.flips: List[RC] = []
        self.undo_stack: List[List[RC]] = []
        self.message = ""
        self.rng = np.random.default_rng(seed)
        self.load_map(0)

    # ----------------------------------------------------------- actions
    def load_map(self, instance_id: int) -> None:
        """Switch to another test map, with no flips."""
        self.instance_id = int(instance_id) % len(self.split)
        self.instance = self.split.instance(self.instance_id, seed=self.seed)
        self.flips, self.undo_stack = [], []
        self.message = f"map {self.instance_id} of {len(self.split)}"
        self._clean = self._evaluate([])
        self.view = self._clean

    def _set(self, flips: Sequence[RC], message: str) -> None:
        self.undo_stack.append(list(self.flips))
        self.flips = [tuple(rc) for rc in flips]
        self.message = message
        self.view = self._evaluate(self.flips)

    def toggle(self, rc: RC) -> bool:
        """Flip one cell, or un-flip it if it was already flipped.

        Returns:
            bool: False if refused (start, goal, outside, or the goal would be
            cut off from the start).
        """
        rc = (int(rc[0]), int(rc[1]))
        h, w = self.instance.map_design.shape
        if not (0 <= rc[0] < h and 0 <= rc[1] < w):
            return False
        if rc in (self.instance.start_rc, self.instance.goal_rc):
            self.message = "start and goal cannot be flipped"
            return False
        flips = [c for c in self.flips if c != rc] if rc in self.flips else self.flips + [rc]
        if not is_reachable(flip_cells(self.instance.map_design, flips),
                            self.instance.start_rc, self.instance.goal_rc):
            self.message = f"refused: flipping {rc} cuts the goal off from the start"
            return False
        verb = "free" if self.instance.map_design[rc] == 0 else "obstacle"
        self._set(flips, f"cell {rc} -> {verb}" if rc not in self.flips else f"cell {rc} restored")
        return True

    def undo(self) -> None:
        """Go back one action."""
        if self.undo_stack:
            self.flips = self.undo_stack.pop()
            self.view = self._evaluate(self.flips)
            self.message = "undone"

    def reset(self) -> None:
        """Back to the clean map."""
        if self.flips:
            self._set([], "reset to the clean map")

    def worst(self, objective: str) -> None:
        """Replace the flips with the single worst cell found by the exhaustive sweep."""
        manifest = self.manifest()
        if manifest is None:
            self.message = "no exhaustive sweep found: run `make sweep` first"
            return
        flag = "is_worst_optimality" if objective == "optimality" else "is_worst_efficiency"
        rows = manifest[(manifest["instance_id"] == self.instance_id)
                        & manifest[flag].to_numpy(dtype=bool, na_value=False)]
        if rows.empty:
            self.message = f"no single cell damages {objective} on this map"
            return
        cell = tuple(int(v) for v in rows.iloc[0]["flip_cells"][0])
        tried = int(((manifest["instance_id"] == self.instance_id)).sum())
        self._set([cell], f"worst single cell for {objective}, out of {tried} tried: {cell}")

    def gradient_step(self, objective: str) -> None:
        """Add one cell chosen by the gradient-guided attack (step 13)."""
        started = time.perf_counter()
        history = greedy_flip_search(
            self.planner, self.instance, budget=1, objective=objective,
            candidate_pool=gradient_pool(self.planner, self.instance, objective, size=64),
            method="gradient", initial=self.flips,
        )
        if not history:
            self.message = "gradient attack found no admissible cell to add"
            return
        cells = [tuple(int(v) for v in rc) for rc in history[0]["flip_cells"]]
        added = [rc for rc in cells if rc not in self.flips]
        self._set(cells, f"gradient attack ({objective}): added {added[0] if added else '-'}"
                         f" — 64 cells tried in {time.perf_counter() - started:.1f} s")

    def random_flip(self) -> None:
        """Flip one uniformly random cell that keeps the goal reachable."""
        cells = candidate_cells(self.instance.map_design, self.instance.start_rc, self.instance.goal_rc)
        for _ in range(200):
            rc = cells[self.rng.integers(len(cells))]
            if rc not in self.flips and self.toggle(rc):
                self.message = "random " + self.message
                return

    def manifest(self):
        """The exhaustive single-cell sweep for this split, if it has been run."""
        if self._manifest is None and self.manifest_path.exists():
            import pandas as pd

            frame = pd.read_parquet(self.manifest_path)
            self._manifest = frame[(frame["method"] == "exhaustive") & (frame["budget"] == 1)
                                   & (frame["seed"] == self.seed)]
        return self._manifest

    # ----------------------------------------------------------- evaluation
    def _evaluate(self, flips: Sequence[RC]) -> Dict[str, object]:
        inst = self.instance
        altered = flip_cells(inst.map_design, flips)
        maps0, starts, goals = inst.to_tensors()
        maps1 = torch.from_numpy(altered.astype(np.float32)).reshape_as(maps0)
        with torch.no_grad():
            cost0 = encode_cost_map(self.planner, maps0, starts, goals)
            cost1 = encode_cost_map(self.planner, maps1, starts, goals)
            out = run_astar_module(self.planner, cost1, starts, goals, maps1)
        optimum = solve_dijkstra(altered, inst.start_rc, inst.goal_rc)
        classical = solve_astar(altered, inst.start_rc, inst.goal_rc)
        path_map = out.paths[0, 0].numpy()
        moves = float(path_map.sum() - 1)
        expanded = float(out.histories[0, 0].numpy().sum())
        swap = run_swap(self.planner, inst, list(flips), RECEPTIVE_RADIUS) if flips else None
        return {
            "map": altered,
            "cost0": cost0[0, 0].numpy(),
            "cost1": cost1[0, 0].numpy(),
            "histories": out.histories[0, 0].numpy(),
            "path": order_path(path_map, inst.start_rc),
            "optimal_path": optimum.path,
            "moves": moves,
            "optimal_moves": float(optimum.cost),
            "expanded": expanded,
            "classical_expanded": float(classical.expansions),
            "opt": moves / float(optimum.cost),
            "exp": expanded / float(classical.expansions),
            "swap": swap,
            "tamper": self.detector.score(inst, altered) if self.detector else None,
        }

    def verdict(self) -> Tuple[str, str]:
        """The project's error rule applied to the current map, as (level, text)."""
        v = self.view
        extra = v["moves"] - v["optimal_moves"]
        if extra > TOL:
            return "fail", f"PLANNER FAILED — route {extra:.0f} moves longer than this map's optimum"
        if v["exp"] > 1 + TOL:
            return "warn", "route optimal, but the search did more work than classical A*"
        return "ok", "route optimal, with less work than classical A*"


# ===================================================================== view
COLORS = {
    "free": (1.0, 1.0, 1.0), "obstacle": (0.17, 0.18, 0.22),
    "expanded": (0.36, 0.62, 0.90, 0.45), "route": "#ff7a1a", "optimal": "#16a394",
    "flip": "#ffd400", "start": "#2fbf4a", "goal": "#e5383b",
    "ok": "#2fbf4a", "warn": "#f2a81d", "fail": "#e5383b",
}


def explain(name: str, values: Dict[str, float], unit: str, floor: float, fmt: str) -> Tuple[str, None, str]:
    """One plain-language line on which change produced the damage.

    `values` holds one metric under the four swap conditions (the bars of the
    panel next to it). The change is attributed to the network when its new
    costs on the old obstacles reproduce at least half of it, to the search
    when the old costs on the new obstacles do. Changes smaller than `floor`
    are called negligible instead of being split into noisy shares.
    """
    base, total = values["C0O0"], values["C1O1"] - values["C0O0"]
    amount = (fmt.format(total) + (f" {unit}" if unit else "")).strip()
    if abs(total) < floor:
        return (f"{name}: barely changed ({amount}).", None, "#444")
    network = (values["C1O0"] - base) / total >= 0.5
    search = (values["C0O1"] - base) / total >= 0.5
    if network and search:
        verdict = "either change alone is enough."
    elif network:
        verdict = "mostly the network: its new costs alone do it."
    elif search:
        verdict = "mostly the search: the old costs fail on the new obstacles."
    else:
        verdict = "neither alone; costs and obstacles only hurt together."
    return (f"{name} {amount}: {verdict}", None, "#444")


class DemoView:
    """Draws a `DemoState` into one matplotlib figure; wires buttons when interactive."""

    def __init__(self, state: DemoState, interactive: bool) -> None:
        import matplotlib.pyplot as plt

        self.state = state
        self.plt = plt
        self.fig = plt.figure(figsize=(16, 9), facecolor="white")
        self.fig.canvas.manager.set_window_title("Neural A* — flip a cell") if interactive else None
        grid = self.fig.add_gridspec(2, 3, left=0.02, right=0.975, top=0.88, bottom=0.14,
                                     wspace=0.22, hspace=0.3, width_ratios=[1.5, 1, 1.05])
        self.ax_map = self.fig.add_subplot(grid[:, 0])
        self.ax_cost = self.fig.add_subplot(grid[0, 1])
        self.ax_delta = self.fig.add_subplot(grid[0, 2])
        self.ax_swap = self.fig.add_subplot(grid[1, 1])
        self.ax_text = self.fig.add_subplot(grid[1, 2])
        self.colorbars = []
        self.buttons = []
        self._add_buttons()
        if interactive:
            self.fig.canvas.mpl_connect("button_press_event", self._on_click)
            self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.draw()

    # ------------------------------------------------------------ controls
    def _add_buttons(self) -> None:
        from matplotlib.widgets import Button

        s = self.state
        spec = [
            ("worst cell: path  [w]", lambda: s.worst("optimality")),
            ("worst cell: effort  [e]", lambda: s.worst("efficiency")),
            ("gradient +1: path  [g]", lambda: s.gradient_step("optimality")),
            ("gradient +1: effort  [h]", lambda: s.gradient_step("efficiency")),
            ("random flip  [r]", s.random_flip),
            ("undo  [u]", s.undo),
            ("reset  [0]", s.reset),
            ("◀ map  [", lambda: s.load_map(s.instance_id - 1)),
            ("map ▶  ]", lambda: s.load_map(s.instance_id + 1)),
        ]
        width, gap, left = 0.098, 0.008, 0.03
        for i, (label, action) in enumerate(spec):
            ax = self.fig.add_axes([left + i * (width + gap), 0.025, width, 0.05])
            button = Button(ax, label, color="#eef1f5", hovercolor="#d7e3f4")
            button.label.set_fontsize(9)
            button.on_clicked(lambda _event, a=action: self._run(a))
            self.buttons.append(button)

    def _run(self, action) -> None:
        self.fig.suptitle("working…", fontsize=13, color="#666")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        action()
        self.draw()

    def _on_click(self, event) -> None:
        if event.inaxes is self.ax_map and event.xdata is not None:
            rc = (int(round(event.ydata)), int(round(event.xdata)))
            self._run(lambda: self.state.toggle(rc))

    def _on_key(self, event) -> None:
        s = self.state
        keys = {
            "w": lambda: s.worst("optimality"), "e": lambda: s.worst("efficiency"),
            "g": lambda: s.gradient_step("optimality"), "h": lambda: s.gradient_step("efficiency"),
            "r": s.random_flip, "u": s.undo, "0": s.reset,
            "[": lambda: s.load_map(s.instance_id - 1), "]": lambda: s.load_map(s.instance_id + 1),
        }
        if event.key in keys:
            self._run(keys[event.key])

    # ------------------------------------------------------------- drawing
    def draw(self) -> None:
        for bar in self.colorbars:
            bar.remove()
        self.colorbars = []
        for ax in (self.ax_map, self.ax_cost, self.ax_delta, self.ax_swap, self.ax_text):
            ax.clear()
        s, v = self.state, self.state.view
        self._draw_map(s, v)
        self._draw_costs(s, v)
        self._draw_swap(v)
        self._draw_text(s, v)
        level, text = s.verdict()
        self.fig.suptitle(text, fontsize=15, fontweight="bold", color=COLORS[level], y=0.975)
        clean = s._clean
        for artist in getattr(self, "_texts", []):
            artist.remove()
        compare = (f"against the clean map: route {v['moves'] - clean['moves']:+.0f} moves, "
                   f"work ×{v['expanded'] / clean['expanded']:.2f} "
                   f"({clean['expanded']:.0f} → {v['expanded']:.0f} cells expanded)"
                   if s.flips else "clean map — nothing flipped yet")
        self._texts = [
            self.fig.text(0.5, 0.925, compare, ha="center", fontsize=11, color="#555"),
            self.fig.text(0.02, 0.095, "▸ " + s.message, fontsize=10.5, color="#1f4e9c",
                          fontweight="bold"),
        ]
        self.fig.canvas.draw_idle()

    def _cells(self, ax, cells, **style) -> None:
        from matplotlib.patches import Rectangle

        for r, c in cells:
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, **style))

    def _draw_map(self, s: DemoState, v: Dict[str, object]) -> None:
        ax = self.ax_map
        grid = v["map"]
        rgb = np.where(grid[..., None] > 0, COLORS["free"], COLORS["obstacle"])
        ax.imshow(rgb, interpolation="nearest")
        layer = np.zeros(grid.shape + (4,))
        layer[v["histories"] > 0] = COLORS["expanded"]
        ax.imshow(layer, interpolation="nearest")
        opt = np.array(v["optimal_path"])
        ax.plot(opt[:, 1], opt[:, 0], color=COLORS["optimal"], lw=7, alpha=0.45,
                solid_capstyle="round", label=f"optimal route ({v['optimal_moves']:.0f} moves)")
        route = np.array(v["path"])
        ax.plot(route[:, 1], route[:, 0], color=COLORS["route"], lw=2.6, marker="o", ms=3,
                label=f"Neural A* route ({v['moves']:.0f} moves)")
        self._cells(ax, s.flips, edgecolor=COLORS["flip"], lw=3)
        sr, sc = s.instance.start_rc
        gr, gc = s.instance.goal_rc
        ax.plot(sc, sr, "o", ms=14, color=COLORS["start"], mec="white", mew=2)
        ax.plot(gc, gr, "*", ms=20, color=COLORS["goal"], mec="white", mew=1.5)
        ax.set_xticks([]), ax.set_yticks([])
        from matplotlib.patches import Patch

        handles, _ = ax.get_legend_handles_labels()
        handles += [Patch(color=COLORS["expanded"], label=f"expanded ({v['expanded']:.0f} cells)"),
                    Patch(fill=False, edgecolor=COLORS["flip"], lw=2, label="flipped cell")]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2,
                  fontsize=9.5, frameon=False)
        ax.set_title(f"test map {s.instance_id} — click a cell to flip it", fontsize=12, loc="left")

    def _draw_costs(self, s: DemoState, v: Dict[str, object]) -> None:
        free = v["map"] > 0
        cost = np.ma.masked_where(~free, v["cost1"])
        image = self.ax_cost.imshow(cost, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        self.ax_cost.set_facecolor("#9aa0a6")
        self._cells(self.ax_cost, s.flips, edgecolor=COLORS["flip"], lw=2)
        for ax in (self.ax_cost, self.ax_delta):
            ax.plot(s.instance.start_rc[1], s.instance.start_rc[0], "o", ms=6, color=COLORS["start"], mec="white")
            ax.plot(s.instance.goal_rc[1], s.instance.goal_rc[0], "*", ms=9, color=COLORS["goal"], mec="white")
        self.ax_cost.set_title("network's cost map (dark = cheap)", fontsize=10.5, loc="left")
        self.colorbars.append(self.fig.colorbar(image, ax=self.ax_cost, fraction=0.046, pad=0.03))

        delta = v["cost1"] - v["cost0"]
        limit = max(float(np.abs(delta).max()), 0.05)
        image = self.ax_delta.imshow(delta, cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
        from matplotlib.patches import Rectangle

        for r, c in s.flips:
            k = RECEPTIVE_RADIUS
            self.ax_delta.add_patch(Rectangle((c - k - 0.5, r - k - 0.5), 2 * k + 1, 2 * k + 1,
                                              fill=False, ls=":", lw=1.2, edgecolor="#333"))
        self._cells(self.ax_delta, s.flips, edgecolor="black", lw=1.8)
        self.ax_delta.set_title("cost change from the flips\n(dotted: the network's 11×11 reach)",
                                fontsize=10.5, loc="left")
        self.colorbars.append(self.fig.colorbar(image, ax=self.ax_delta, fraction=0.046, pad=0.03,
                                                format="%.2f"))
        if not s.flips:
            self.ax_delta.text(0.5, 0.5, "no cell flipped yet", ha="center", va="center",
                               color="#777", fontsize=11, transform=self.ax_delta.transAxes)
        for ax in (self.ax_cost, self.ax_delta):
            ax.set_xticks([]), ax.set_yticks([])

    def _draw_swap(self, v: Dict[str, object]) -> None:
        ax = self.ax_swap
        swap = v["swap"]
        ax.set_title("where the damage comes from\n(the search rerun four ways)", fontsize=10.5, loc="left")
        if swap is None:
            ax.text(0.5, 0.5, "flip a cell to decompose its effect", ha="center", va="center",
                    color="#777", fontsize=11, transform=ax.transAxes)
            ax.axis("off")
            return
        labels = ["clean map", "obstacles changed,\nold costs (search)",
                  "costs changed,\nold obstacles (network)", "both changed\n(what happens)"]
        keys = ["C0O0", "C0O1", "C1O0", "C1O1"]
        y = np.arange(4)[::-1]
        opt = [swap.opt[k] for k in keys]
        exp = [swap.exp[k] for k in keys]
        ax.barh(y + 0.19, opt, 0.36, color=COLORS["route"], label="route length ÷ optimum")
        ax.barh(y - 0.19, exp, 0.36, color=COLORS["expanded"][:3], label="cells expanded ÷ classical A*")
        for yy, a, b in zip(y, opt, exp):
            ax.text(a + 0.02, yy + 0.19, f"{a:.2f}", va="center", fontsize=8.5)
            ax.text(b + 0.02, yy - 0.19, f"{b:.2f}", va="center", fontsize=8.5)
        ax.axvline(1, color="#555", lw=0.8, ls="--")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.set_xlim(0, max(max(opt), max(exp), 1.0) * 1.22)
        ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.4, -0.1), ncol=2, frameon=False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    def _draw_text(self, s: DemoState, v: Dict[str, object]) -> None:
        import textwrap

        ax = self.ax_text
        ax.axis("off")
        clean = s._clean
        lines = [
            ("This map, as altered" if s.flips else "This map (clean)", "bold", "#222"),
            (f"route: {v['moves']:.0f} moves vs optimum {v['optimal_moves']:.0f}   (Opt {v['opt']:.3f})",
             None, "#222"),
            (f"work: {v['expanded']:.0f} cells vs classical A* {v['classical_expanded']:.0f}   "
             f"(Exp {v['exp']:.2f})", None, "#222"),
            ("", None, None),
        ]
        swap = v["swap"]
        if swap is not None:
            lines.append(("Why", "bold", "#222"))
            if swap.optimum_changed:
                lines.append((f"The flips changed the true optimum ({clean['optimal_moves']:.0f} → "
                              f"{v['optimal_moves']:.0f} moves); the planner must adapt to that.", None, "#b26b00"))
            lines.append(explain("Route", swap.path_cost, "moves", 0.5, "{:+.0f}"))
            lines.append(explain("Work vs classical A*", swap.exp, "", 0.05, "{:+.2f}"))
        if v["tamper"] is not None:
            control, combined = v["tamper"]
            seen = "" if s.instance_id >= TamperDetector.TRAIN_MAPS else " (map used to fit it)"
            lines += [("", None, None), (f"Tamper detector{seen}", "bold", "#222"),
                      (f"map-only network: P(altered) {control:.2f}", None, "#444"),
                      (f"+ planner's cost map: P(altered) {combined:.2f}", None,
                       COLORS["fail"] if combined > 0.5 else COLORS["ok"])]
        y = 1.0
        for text, weight, color in lines:
            for piece in textwrap.wrap(text, 50) or [""]:
                ax.text(0.0, y, piece, fontsize=10.5 if weight else 10, fontweight=weight or "normal",
                        color=color or "#222", transform=ax.transAxes, va="top")
                y -= 0.075

    def save(self, path: Path) -> None:
        self.fig.savefig(path, dpi=90)


def run_actions(state: DemoState, view: DemoView, actions: Sequence[str], out_dir: Optional[Path],
                gif: Optional[Path] = None, frame_ms: int = 2200) -> None:
    """Apply scripted actions, saving a snapshot after each (headless mode).

    Actions: ``map:N``, ``flip:R,C``, ``worst:opt|eff``, ``grad:opt|eff``,
    ``random``, ``undo``, ``reset``. With `gif`, every state is also a frame
    of an animation — a recorded walkthrough to fall back on if a live demo
    is not possible.
    """
    from PIL import Image

    objective = {"opt": "optimality", "eff": "efficiency"}
    frames = []

    def snapshot(name: str) -> None:
        if out_dir:
            view.save(out_dir / name)
        if gif:
            view.fig.canvas.draw()
            frames.append(Image.fromarray(np.asarray(view.fig.canvas.buffer_rgba())).convert("RGB"))

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    snapshot("step_00_start.png")
    for i, action in enumerate(actions, start=1):
        name, _, arg = action.partition(":")
        if name == "map":
            state.load_map(int(arg))
        elif name == "flip":
            state.toggle(tuple(int(x) for x in arg.split(",")))
        elif name == "worst":
            state.worst(objective[arg])
        elif name == "grad":
            state.gradient_step(objective[arg])
        elif name in ("random", "undo", "reset"):
            getattr(state, {"random": "random_flip"}.get(name, name))()
        else:
            raise ValueError(f"unknown action {action!r}")
        view.draw()
        print(f"{action:>14}: {state.verdict()[1]} | {state.message}")
        snapshot(f"step_{i:02d}_{name}.png")
    if gif and frames:
        gif.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(gif, save_all=True, append_images=frames[1:], duration=frame_ms, loop=0)
        print(f"wrote {gif} ({len(frames)} frames)")


def main() -> int:
    """Open the interactive window, or run scripted actions headless."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map", type=int, default=0, help="test map to start on")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--detector", action="store_true",
                        help="show step 12's tamper detector (fitted once, ~2 min, then cached)")
    parser.add_argument("--actions", default=None, help="headless: space-separated actions")
    parser.add_argument("--snapshots", type=Path, default=None, help="headless: save a PNG per action here")
    parser.add_argument("--gif", type=Path, default=None, help="headless: record the actions as an animation")
    args = parser.parse_args()

    import matplotlib

    if args.actions is not None:
        matplotlib.use("Agg")
    config.set_global_seeds(args.seed)
    planner = load_pretrained_neural_astar(checkpoint_path=args.checkpoint)
    detector = TamperDetector(planner, seed=args.seed) if args.detector else None
    state = DemoState(planner, seed=args.seed, detector=detector)
    state.load_map(args.map)
    view = DemoView(state, interactive=args.actions is None)
    if args.actions is not None:
        run_actions(state, view, args.actions.split(), args.snapshots, args.gif)
        return 0
    view.plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
