"""Interactive demo: flip cells of a maze and watch Neural A* respond, in the browser.

Runs a small local web server (standard library only) and opens the page.
Click any cell to flip it (free <-> obstacle); every view is recomputed by the
real pretrained planner:

* the maze, with the cells Neural A* expanded, its route, and the exact
  optimal route on the map as it now stands; or, by switching layers, the
  network's cost map or how the flips changed it (only cells within the
  network's 11 x 11 reach of a flip can move);
* the verdict by the project's error rule: the planner fails only if its route
  is longer than the *altered* map's optimum; replanning because the map
  changed is the correct response;
* where the damage comes from: the search rerun with only the new costs and
  with only the new obstacles (the 2x2 swap of step 7);
* optionally, step 12's tamper detector.

Actions: worst single cell for path / effort from the exhaustive sweep, one
round of the gradient-guided attack of step 13, a random flip, undo, reset,
previous / next map. The page lists the keyboard shortcuts.

Usage:
    python scripts/demo.py                      # opens http://127.0.0.1:8765/
    python scripts/demo.py --map 63 --detector
    python scripts/demo.py --actions "worst:opt grad:opt"   # headless, prints verdicts
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


def order_path(path_map: np.ndarray, start: RC, goal: Optional[RC] = None) -> List[RC]:
    """Order the cells of a binary path map from start to goal, for drawing.

    The search returns its route as a set of marked cells, not a sequence.
    Walking greedily from the start to "the next neighbour" can jump between
    two marked cells that touch but are not consecutive on the route, and a
    line drawn that way cuts across obstacles that the route itself goes
    around. Instead this takes the shortest chain of 8-neighbouring marked
    cells from start to goal (breadth-first search restricted to the marked
    cells), so every drawn segment joins two cells the route really steps
    between.

    Args:
        path_map: ``(H, W)`` binary map of the route.
        start: The start cell.
        goal: The goal cell; when omitted, the marked cell farthest from the
            start along the chain.

    Returns:
        list[tuple[int, int]]: The route in order, start first.
    """
    from collections import deque

    cells = {tuple(int(v) for v in rc) for rc in np.argwhere(path_map > 0)}
    start = tuple(int(v) for v in start)
    cells.add(start)
    parent: Dict[RC, Optional[RC]] = {start: None}
    queue = deque([start])
    last = start
    while queue:
        r, c = last = queue.popleft()
        if goal is not None and last == tuple(goal):
            break
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nxt = (r + dr, c + dc)
                if nxt in cells and nxt not in parent:
                    parent[nxt] = last
                    queue.append(nxt)
    end = tuple(goal) if goal is not None and tuple(goal) in parent else last
    order = []
    while end is not None:
        order.append(end)
        end = parent[end]
    return order[::-1]


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
        self.message = f"Test map {self.instance_id} (of 0–{len(self.split) - 1}). Click a cell to flip it."
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
            "path": order_path(path_map, inst.start_rc, inst.goal_rc),
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
            moves = "move" if round(extra) == 1 else "moves"
            return "fail", f"The planner failed: its route is {extra:.0f} {moves} longer than the shortest"
        if v["exp"] > 1 + TOL:
            return "warn", "Shortest route found, but with more work than classical A*"
        return "ok", "Shortest route found, with less work than classical A*"


def explain(values: Dict[str, float], floor: float) -> str:
    """Which change produced the damage, as one plain-language phrase.

    `values` holds one metric under the four swap conditions. The change is
    attributed to the network when its new costs on the old obstacles
    reproduce at least half of it, to the search when the old costs on the new
    obstacles do. Changes smaller than `floor` are called negligible instead
    of being split into noisy shares.
    """
    base, total = values["C0O0"], values["C1O1"] - values["C0O0"]
    if abs(total) < floor:
        return "barely changed"
    network = (values["C1O0"] - base) / total >= 0.5
    search = (values["C0O1"] - base) / total >= 0.5
    if network and search:
        return "either change alone is enough"
    if network:
        return "the network: its new costs alone do it"
    if search:
        return "the search: the old costs fail on the new obstacles"
    return "neither alone: costs and obstacles only hurt together"


def snapshot(state: DemoState) -> Dict[str, object]:
    """Everything the page draws, as JSON-ready values."""
    v, clean, inst = state.view, state._clean, state.instance
    level, headline = state.verdict()
    swap = v["swap"]
    out: Dict[str, object] = {
        "map_id": state.instance_id,
        "n_maps": len(state.split),
        "grid": v["map"].astype(int).tolist(),
        "start": list(inst.start_rc),
        "goal": list(inst.goal_rc),
        "flips": [list(rc) for rc in state.flips],
        "expanded": (v["histories"] > 0).astype(int).tolist(),
        "route": [list(rc) for rc in v["path"]],
        "optimal_route": [list(map(int, rc)) for rc in v["optimal_path"]],
        "cost": np.round(v["cost1"], 3).tolist(),
        "cost_change": np.round(v["cost1"] - v["cost0"], 3).tolist(),
        "level": level,
        "headline": headline,
        "message": state.message,
        "now": {"moves": v["moves"], "optimum": v["optimal_moves"], "expanded": v["expanded"],
                "classical": v["classical_expanded"]},
        "clean": {"moves": clean["moves"], "optimum": clean["optimal_moves"],
                  "expanded": clean["expanded"], "classical": clean["classical_expanded"]},
        "swap": None,
        "tamper": None,
        "has_sweep": state.manifest() is not None,
    }
    if swap is not None:
        out["swap"] = {
            "moves": swap.path_cost, "expanded": swap.expansions,
            "optimum": {"O0": swap.references["O0"][0], "O1": swap.references["O1"][0]},
            "classical": {"O0": swap.references["O0"][1], "O1": swap.references["O1"][1]},
            "optimum_changed": swap.optimum_changed,
            "route_reading": explain(swap.path_cost, 0.5),
            "work_reading": explain(swap.exp, 0.05),
        }
    if v["tamper"] is not None:
        out["tamper"] = {"map_only": v["tamper"][0], "with_planner": v["tamper"][1],
                         "fitted_on_this_map": state.instance_id < TamperDetector.TRAIN_MAPS}
    return out


def apply_action(state: DemoState, name: str, arg: object = None) -> None:
    """Run one named action on the state (shared by the web page and headless mode)."""
    objective = {"opt": "optimality", "eff": "efficiency"}
    if name == "flip":
        state.toggle(tuple(int(x) for x in arg))
    elif name == "map":
        state.load_map(int(arg))
    elif name == "worst":
        state.worst(objective[arg])
    elif name == "grad":
        state.gradient_step(objective[arg])
    elif name == "random":
        state.random_flip()
    elif name == "undo":
        state.undo()
    elif name == "reset":
        state.reset()
    else:
        raise ValueError(f"unknown action {name!r}")


def png_data_url(image: np.ndarray) -> str:
    """A ``(H, W, 3)`` image in [0, 1] or uint8 as an inline PNG for the page."""
    import base64
    import io

    from PIL import Image

    array = image if image.dtype == np.uint8 else (np.clip(image, 0, 1) * 255).round().astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


class WarcraftState:
    """The demo on WarCraft terrain pictures (step 11).

    Each map is a 96 x 96 picture of 12 x 12 terrain tiles with a hidden true
    cost each (0.8 to 9.2); the planner sees only the picture. Two kinds of
    change are offered: repainting a tile with a real tile of another terrain
    from the training maps (the true costs change, so the shortest route may
    move), and an invisible pixel attack (every pixel nudged by at most
    eps/255; the true costs do not change at all).
    """

    def __init__(self, seed: int = 0) -> None:
        import glob

        from src import warcraft as wc

        self.wc = wc
        checkpoints = sorted(glob.glob(str(config.RESULTS_DIR / "models" / "warcraft_seed0" / "**" / "*.ckpt"),
                                       recursive=True))
        if not checkpoints:
            raise FileNotFoundError("no WarCraft model: run `make warcraft-train`")
        self.planner = wc.load_warcraft_planner(Path(checkpoints[-1]))
        self.test = wc.load_warcraft_split("test")
        self.train = wc.load_warcraft_split("train")
        self.pool = wc.donor_pool(self.train.weights)
        self.rng = np.random.default_rng(seed)
        outputs = config.RESULTS_DIR / "warcraft" / "warcraft_seed0" / "test_outputs.npz"
        self.solved = (np.load(outputs)["opt"] <= 1 + wc.OPT_TOL) if outputs.exists() else None
        # One real training tile per terrain class, shown as the paint palette.
        # The dataset does not name its terrains, so neither does the demo.
        self.swatches = []
        for cls in wc.TERRAIN_COSTS:
            m, r, c = (int(v) for v in self.pool[cls][0])
            self.swatches.append({"cost": cls, "image": png_data_url(self.train.images[m][wc.tile_slice((r, c))])})
        sweep = config.RESULTS_DIR / "warcraft" / "warcraft_seed0" / "manifest.parquet"
        self.sweep = None
        if sweep.exists():
            import pandas as pd

            self.sweep = pd.read_parquet(sweep, columns=["instance_id", "flip_cells", "new_weight",
                                                       "donors", "opt_C1", "opt_C0_on_W1"])
        # Open on a map the planner solves: an attack is only meaningful where
        # the untouched planner was right, as in step 11.
        self.load_map(0)
        if self.solved is not None and not self.solved[0]:
            self.next_solved(1)

    def load_map(self, instance_id: int) -> None:
        self.instance_id = int(instance_id) % len(self.test)
        self.image = self.test.images[self.instance_id].copy()
        self.weights = self.test.weights[self.instance_id].copy()
        self.delta: Optional[np.ndarray] = None
        self.eps = 0
        self.edits: List[Tuple[RC, float]] = []
        self.undo_stack: List[tuple] = []
        self.message = f"WarCraft test map {self.instance_id}. Pick a terrain, then click a tile to repaint it."
        self.clean = self._evaluate()
        self.view = self.clean

    def next_solved(self, step: int) -> None:
        """Jump to the next map the planner solves optimally when untouched."""
        if self.solved is None:
            self.load_map(self.instance_id + step)
            return
        i = self.instance_id
        for _ in range(len(self.test)):
            i = (i + step) % len(self.test)
            if self.solved[i]:
                break
        self.load_map(i)
        self.message = f"WarCraft test map {i}: one the planner solves when nothing is changed."

    def _push(self) -> None:
        self.undo_stack.append((self.image.copy(), self.weights.copy(), self.delta, self.eps, list(self.edits)))

    def undo(self) -> None:
        if self.undo_stack:
            self.image, self.weights, self.delta, self.eps, self.edits = self.undo_stack.pop()
            self.view = self._evaluate()
            self.message = "undone"

    def reset(self) -> None:
        if self.edits or self.delta is not None:
            self._push()
            self.image = self.test.images[self.instance_id].copy()
            self.weights = self.test.weights[self.instance_id].copy()
            self.delta, self.eps, self.edits = None, 0, []
            self.view = self._evaluate()
            self.message = "reset to the original picture"

    def paint(self, rc: RC, cls: float) -> None:
        """Replace tile `rc` with a real tile of terrain `cls` from a training map."""
        rc = (int(rc[0]), int(rc[1]))
        if rc in (self.wc.START_RC, self.wc.GOAL_RC):
            self.message = "start and goal tiles cannot be repainted"
            return
        if any(e[0] == rc for e in self.edits):
            # Clicking a repainted tile puts the original back, as a click on a
            # flipped maze cell does.
            self._push()
            original = self.test.images[self.instance_id]
            sl = self.wc.tile_slice(rc)
            self.image = self.image.copy()
            self.image[sl] = original[sl]
            self.weights = self.weights.copy()
            self.weights[rc] = self.test.weights[self.instance_id][rc]
            self.edits = [e for e in self.edits if e[0] != rc]
            self.view = self._evaluate()
            self.message = f"tile {rc} restored"
            return
        cls = min(self.wc.TERRAIN_COSTS, key=lambda c: abs(c - float(cls)))
        donor = self.wc.draw_donor(self.pool, cls, self.rng)
        self._push()
        self.image, self.weights = self.wc.transplant_tiles(
            self.image, self.weights, [rc], [self.train.images[donor[0]]], [(donor[1], donor[2])], [cls])
        self.edits = [e for e in self.edits if e[0] != rc] + [(rc, cls)]
        self.view = self._evaluate()
        self.message = f"tile {rc} repainted with terrain of true cost {cls}"

    def worst(self) -> None:
        """The single tile repaint that hurt the planner most, from the exhaustive sweep."""
        if self.sweep is None:
            self.message = "no WarCraft sweep found: run `make warcraft`"
            return
        rows = self.sweep[self.sweep["instance_id"] == self.instance_id]
        if rows.empty:
            self.message = "the sweep covered test maps 0-299 only; pick one of those"
            return
        row = rows.loc[rows["opt_C1"].idxmax()]
        rc = tuple(int(v) for v in row["flip_cells"][0])
        donor = tuple(int(v) for v in row["donors"][0])
        cls = float(row["new_weight"][0])
        self.reset_silently()
        self._push()
        self.image, self.weights = self.wc.transplant_tiles(
            self.image, self.weights, [rc], [self.train.images[donor[0]]], [(donor[1], donor[2])], [cls])
        self.edits = [(rc, cls)]
        self.view = self._evaluate()
        self.message = (f"worst single repaint out of {len(rows)} tried: tile {rc}, "
                        f"terrain of true cost {cls}")

    def reset_silently(self) -> None:
        self.image = self.test.images[self.instance_id].copy()
        self.weights = self.test.weights[self.instance_id].copy()
        self.delta, self.eps, self.edits = None, 0, []

    def pixel_attack(self, eps_255: int, kind: str) -> None:
        """Nudge every pixel by at most eps/255: gradient-guided (PGD) or random signs."""
        from scripts.warcraft_study import pgd

        eps = int(eps_255) / 255.0
        self._push()
        base = self.image.astype(np.float32) / 255.0
        if kind == "pgd":
            target = np.zeros((1, self.wc.GRID, self.wc.GRID), dtype=np.float32)
            for rc in self.wc.optimal_solution(self.weights).path:
                target[0][rc] = 1.0
            attacked = pgd(self.planner, self.image[None], target, eps, 10)[0]
        else:
            attacked = np.clip(base + self.rng.choice([-eps, eps], size=base.shape), 0, 1)
        self.delta = (attacked - base).astype(np.float32)
        self.eps = int(eps_255)
        self.view = self._evaluate()
        what = "gradient attack" if kind == "pgd" else "random noise"
        self.message = f"{what}: every pixel moved by at most {eps_255}/255 — the true costs did not change"

    def _evaluate(self) -> Dict[str, object]:
        wc = self.wc
        x = self.image.astype(np.float32) / 255.0
        if self.delta is not None:
            x = np.clip(x + self.delta, 0, 1)
        costs, paths, hists = wc.plan(self.planner, x[None])
        sc = wc.score(self.weights[None], paths, hists, costs)
        from scipy.stats import spearmanr

        return {
            "input": x,
            "cost": costs[0],
            "route": order_path(paths[0], wc.START_RC, wc.GOAL_RC),
            "expanded": hists[0] > 0,
            "optimal_route": [tuple(int(v) for v in rc) for rc in wc.optimal_solution(self.weights).path],
            "exact_route": order_path(wc.exact_path_on(costs[0]), wc.START_RC, wc.GOAL_RC),
            "path_cost": float(sc["path_cost"][0]), "optimum": float(sc["optimum"][0]),
            "exact_cost": float(sc["opt_exact"][0] * sc["optimum"][0]),
            "expanded_n": float(sc["expansions"][0]), "classical": float(sc["classical_expansions"][0]),
            "spearman": float(spearmanr(costs[0].ravel(), self.weights.ravel()).correlation),
        }

    def context(self) -> str:
        """How good this planner is untouched, so a failure can be read correctly."""
        if self.solved is None:
            return "This planner is weak on terrain maps: untouched, it finds the cheapest route on about a third of them."
        n, k = len(self.solved), int(self.solved.sum())
        return (f"This planner is weak on terrain maps: untouched, it finds the exact cheapest route on {k} of "
                f"{n} test maps ({k / n:.0%}). The arrows visit only those, so a failure after a change is "
                f"caused by the change.")

    def verdict(self) -> Tuple[str, str]:
        v = self.view
        extra = v["path_cost"] - v["optimum"]
        if extra > 1e-6 and not self.edits and self.delta is None:
            return "warn", f"Untouched, the planner's route already costs {extra / v['optimum']:.0%} more than the cheapest"
        if extra > 1e-6:
            return "fail", f"The planner failed: its route costs {extra / v['optimum']:.0%} more than the cheapest"
        return "ok", "Cheapest route found"

    def snapshot(self) -> Dict[str, object]:
        v, c = self.view, self.clean
        level, headline = self.verdict()
        cost = v["cost"]
        span = float(cost.max() - cost.min()) or 1.0
        out = {
            "mode": "warcraft",
            "map_id": self.instance_id,
            "n_maps": len(self.test),
            "image": png_data_url(v["input"]),
            "perturbation": png_data_url(0.5 + self.delta / (2 * self.eps / 255.0)) if self.delta is not None else None,
            "eps": self.eps,
            "weights": self.weights.tolist(),
            "cost": np.round((cost - cost.min()) / span, 3).tolist(),
            "cost_raw": np.round(cost, 3).tolist(),
            # Colour by rank: the network's costs span two orders of magnitude,
            # and a linear ramp would leave most tiles the same pale shade.
            "cost_rank": np.round(np.argsort(np.argsort(cost.ravel())).reshape(cost.shape) / (cost.size - 1), 3).tolist(),
            "route": [list(rc) for rc in v["route"]],
            "optimal_route": [list(rc) for rc in v["optimal_route"]],
            "exact_route": [list(rc) for rc in v["exact_route"]],
            "expanded": v["expanded"].astype(int).tolist(),
            "edits": [[list(rc), cls] for rc, cls in self.edits],
            "start": list(self.wc.START_RC), "goal": list(self.wc.GOAL_RC),
            "terrains": self.swatches,
            "level": level, "headline": headline, "message": self.message,
            "now": {k: v[k] for k in ("path_cost", "optimum", "exact_cost", "expanded_n", "classical", "spearman")},
            "clean": {k: c[k] for k in ("path_cost", "optimum", "exact_cost", "expanded_n", "classical", "spearman")},
            "has_sweep": self.sweep is not None and self.instance_id < 300,
            "solved_clean": bool(c["path_cost"] <= c["optimum"] + 1e-6),
            "context": self.context(),
        }
        return out


class Demo:
    """Both demos behind one server; the page switches between them."""

    def __init__(self, maze: DemoState, seed: int = 0) -> None:
        self.maze = maze
        self.seed = seed
        self.warcraft: Optional[WarcraftState] = None
        self.mode = "maze"
        self.warcraft_error = ""

    def snapshot(self, mode: Optional[str] = None) -> Dict[str, object]:
        """The state of one map type: the one the page names, else the default."""
        mode = mode or self.mode
        if mode == "warcraft" and self.warcraft is not None:
            return self.warcraft.snapshot()
        out = snapshot(self.maze)
        out["mode"] = "maze"
        if self.warcraft_error:
            out["message"] = self.warcraft_error
            self.warcraft_error = ""
        return out

    def apply(self, name: str, arg: object = None, mode: Optional[str] = None) -> str:
        """Run one action on the map type the page is showing; returns that type.

        The page sends its own map type with every action. Keeping it per
        request, rather than as server state, means a second browser tab (or a
        stale one from an earlier `make demo`) can never switch the map type
        under another tab, nor send a maze click to the terrain maps.
        """
        if name == "mode":
            if arg == "warcraft" and self.warcraft is None:
                try:
                    self.warcraft = WarcraftState(self.seed)
                except (FileNotFoundError, OSError) as error:
                    self.warcraft_error = f"WarCraft is not available here: {error}. Run `make warcraft-data` and `make warcraft-train`."
                    return
            self.mode = "warcraft" if arg == "warcraft" else "maze"
            return self.mode
        mode = mode or self.mode
        if mode != "warcraft" or self.warcraft is None:
            apply_action(self.maze, name, arg)
            return "maze"
        w = self.warcraft
        if name == "paint":
            w.paint(arg[:2], arg[2])
        elif name == "map":
            w.load_map(int(arg))
        elif name == "solved":
            w.next_solved(int(arg))
        elif name == "worst":
            w.worst()
        elif name == "pixels":
            w.pixel_attack(int(arg[0]), arg[1])
        elif name in ("undo", "reset"):
            getattr(w, name)()
        else:
            raise ValueError(f"unknown action {name!r}")
        return "warcraft"


PAGE = Path(__file__).with_name("demo_page.html")


def make_server(state: "Demo", port: int):
    """A single-threaded local HTTP server: the page, its state, and its actions.

    One request at a time on purpose: the planner is not shared across threads,
    and one person drives the demo.
    """
    import json
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: bytes, kind: str, code: int = 200) -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            from urllib.parse import parse_qs, urlsplit

            url = urlsplit(self.path)
            mode = parse_qs(url.query).get("mode", [""])[0] or None
            if url.path in ("/", "/index.html"):
                self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/state":
                self._send(json.dumps(state.snapshot(mode)).encode(), "application/json")
            else:
                self._send(b"not found", "text/plain", 404)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/action":
                self._send(b"not found", "text/plain", 404)
                return
            length = int(self.headers.get("Content-Length", 0))
            mode = None
            try:
                request = json.loads(self.rfile.read(length) or b"{}")
                mode = request.get("mode")
                mode = state.apply(request.get("name", ""), request.get("arg"), mode)
            except (ValueError, KeyError, TypeError, IndexError) as error:
                target = state.warcraft if mode == "warcraft" and state.warcraft else state.maze
                target.message = f"could not do that: {error}"
            self._send(json.dumps(state.snapshot(mode)).encode(), "application/json")

        def log_message(self, *args) -> None:  # keep the terminal quiet
            pass

    return HTTPServer(("127.0.0.1", port), Handler)


def main() -> int:
    """Serve the demo page locally, or run scripted actions headless."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map", type=int, default=0, help="test map to start on")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--detector", action="store_true",
                        help="show step 12's tamper detector (fitted once, ~1 min, then cached)")
    parser.add_argument("--warcraft", action="store_true", help="open on the WarCraft terrain maps")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    parser.add_argument("--actions", default=None,
                        help="headless: space-separated actions, e.g. 'worst:opt grad:opt flip:5,7'")
    args = parser.parse_args()

    config.set_global_seeds(args.seed)
    planner = load_pretrained_neural_astar(checkpoint_path=args.checkpoint)
    detector = TamperDetector(planner, seed=args.seed) if args.detector else None
    state = DemoState(planner, seed=args.seed, detector=detector)
    state.load_map(args.map)

    if args.actions is not None:
        for action in args.actions.split():
            name, _, arg = action.partition(":")
            apply_action(state, name, arg.split(",") if name == "flip" else (arg or None))
            print(f"{action:>14}: {state.verdict()[1]} | {state.message}")
        return 0

    demo = Demo(state, seed=args.seed)
    if args.warcraft:
        demo.apply("mode", "warcraft")
    server = make_server(demo, args.port)
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"demo running at {url}  (Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        import webbrowser

        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
