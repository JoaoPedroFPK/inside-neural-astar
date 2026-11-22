"""The interpretability half of the browser demo: what the cost map encodes.

Everything here runs live on the map the page is showing, clean or altered,
with the project's own methods and fitted the way the experiments fit them:

* **Surrogate models and the sufficiency test (step 5).** Gradient-boosting
  surrogates of the encoder's cost map, fitted on the 800 training maps from
  the interpretable feature library (step 4): one on geometry only (distance to
  the goal and to walls, hypothesis H1 as stated) and one on every feature,
  including where the optimal routes run. Each surrogate's prediction is fed
  to the *same* search the encoder feeds, so a surrogate is scored by how it
  plans, not only by how closely it copies the network.
* **Feature correlations (step 4)** of the cost map with single features.
* **Attribution (step 8).** Integrated gradients and Grad-CAM for one cost
  value, checked against the encoder's exact 11 x 11 receptive field, plus the
  randomisation sanity check: the same attribution on an encoder with random
  weights should look different.
* **WarCraft (step 11).** The position pattern the network's costs follow,
  ablated by removing it before the search, and linear probes that read the
  true terrain cost off the last hidden layer and off the output.
"""

from __future__ import annotations

import pickle
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.attribution import (  # noqa: E402
    attribution_radius,
    grad_cam,
    integrated_gradients,
    randomize_encoder,
)
from src.exact_solver import distance_field  # noqa: E402
from src.features import compute_instance_features, on_solver_path  # noqa: E402
from src.grid import RC  # noqa: E402
from src.planner import run_astar_module  # noqa: E402
from src.surrogate import FEATURE_GROUPS, build_design_matrix, fit_and_score  # noqa: E402

#: The two surrogates of step 5 the demo shows, by feature group.
SURROGATES = ("geometric_h1", "all")

#: Single features whose correlation with the cost map step 4 reported.
CORRELATES = {
    "on_any_optimal_path": "On an optimal route",
    "geodesic_to_goal": "Distance to the goal",
    "chebyshev_to_goal": "The search's own heuristic",
    "dist_to_obstacle_chebyshev": "Distance to the nearest wall",
}

WARCRAFT_CACHE = config.RESULTS_DIR / "cache" / "demo_warcraft_explain.pkl"


def r2_on(y: np.ndarray, prediction: np.ndarray) -> float:
    """Share of the variance of `y` that `prediction` explains (can be negative)."""
    spread = float(((y - y.mean()) ** 2).sum())
    return float("nan") if spread == 0 else 1.0 - float(((y - prediction) ** 2).sum()) / spread


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation, NaN when either side is constant."""
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")


class MazeExplainer:
    """Step 5's surrogates and step 8's attribution, applied to one map at a time."""

    def __init__(self, planner, split, seed: int = 0) -> None:
        self.planner = planner
        self.split = split
        train, test = build_design_matrix("train"), build_design_matrix("test")
        # Same fits as step 5 (gradient boosting, same seed), so the test R² is
        # step 5's: 0.240 for geometry, 0.408 for every feature.
        self.fits = {g: fit_and_score(train, test, g, kind="gbm", seed=seed) for g in SURROGATES}
        self.randomized = randomize_encoder(planner, seed=seed)

    def features(self, instance, design: np.ndarray, clean: bool) -> Dict[str, np.ndarray]:
        """Step 4's feature library for the map as it stands.

        On the clean map the reference route is the dataset's (the planner's
        training target), exactly as step 4 stored it. An altered map has no
        dataset route, so the exact solver's optimal route stands in for it.
        """
        goal, start = instance.goal_rc, instance.start_rc
        current = replace(instance, map_design=design, opt_dist=distance_field(design, goal))
        if clean:
            reference = self.split.reference_trajectory(instance.instance_id, start)
        else:
            reference = on_solver_path(design, start, goal).astype(np.float32)
            reference[goal] = 0.0  # the dataset's convention: start marked, goal not
        return compute_instance_features(current, reference_path=reference)

    def explain(self, instance, design: np.ndarray, cost: np.ndarray, clean: bool) -> Dict[str, object]:
        """Surrogate cost maps, how each plans, and the feature correlations."""
        f = self.features(instance, design, clean)
        mask = (f["is_free"] > 0) & (f["reachable"] > 0)
        y = cost[mask].astype(np.float64)
        _, starts, goals = instance.to_tensors()
        obstacles = torch.from_numpy(design.astype(np.float32)).reshape_as(starts)
        out: Dict[str, object] = {"surrogates": {}, "correlations": {}}
        for group, fit in self.fits.items():
            x = np.stack([f[name][mask] for name in FEATURE_GROUPS[group]], axis=1).astype(np.float64)
            prediction = np.clip(fit.model.predict(x), 1e-6, 1.0 - 1e-6)
            # Cells off the start's component are never expanded, so their
            # value is inert; the mean keeps the map rectangular (as step 5).
            grid = np.full(design.shape, prediction.mean(), dtype=np.float32)
            grid[mask] = prediction
            with torch.no_grad():
                run = run_astar_module(self.planner, torch.from_numpy(grid).reshape_as(starts),
                                       starts, goals, obstacles)
            out["surrogates"][group] = {
                "cost": grid,
                "r2_this_map": r2_on(y, prediction),
                "r2_test_split": fit.r2,
                "moves": float(run.paths[0, 0].sum() - 1),
                "expanded": float(run.histories[0, 0].sum()),
                "path_map": run.paths[0, 0].numpy(),
            }
        out["residual"] = np.where(mask, cost - out["surrogates"]["all"]["cost"], 0.0)
        out["mask"] = mask
        for name in CORRELATES:
            out["correlations"][name] = pearson(f[name][mask].astype(np.float64), y)
        return out

    def attribute(self, instance, design: np.ndarray, target: RC) -> Dict[str, object]:
        """Step 8's three attribution maps for the cost value at `target`."""
        current = replace(instance, map_design=design)
        ig = integrated_gradients(self.planner, current, target, baseline="free", steps=64)
        cam = grad_cam(self.planner, current, target)
        rand = integrated_gradients(self.randomized, current, target, baseline="free", steps=64)
        return {
            "target": target,
            "value": ig.output_value,
            "ig": ig.values[0],
            "cam": cam.values[0],
            "random": rand.values[0],
            "ig_inside": attribution_radius(ig)["share_within_5"],
            "cam_inside": attribution_radius(cam)["share_within_5"],
            "random_inside": attribution_radius(rand)["share_within_5"],
            # Step 8's statistic: both input channels, trained vs random weights.
            "random_r": pearson(ig.values.ravel(), rand.values.ravel()),
            "completeness_error": ig.completeness_error,
            "baseline_value": ig.baseline_output,
        }


class WarcraftExplainer:
    """Step 11's position pattern and linear probes, for the terrain maps.

    Fitted once on the first 300 training maps for the probes (step 11's
    setting) and the first 2 000 for the position pattern, then cached.
    """

    PROBE_MAPS = 300
    POSITION_MAPS = 2000

    def __init__(self, planner, train, seed: int = 0) -> None:
        self.planner = planner
        if WARCRAFT_CACHE.exists():
            parts = pickle.loads(WARCRAFT_CACHE.read_bytes())
        else:
            parts = self._fit(train, seed)
            WARCRAFT_CACHE.parent.mkdir(parents=True, exist_ok=True)
            WARCRAFT_CACHE.write_bytes(pickle.dumps(parts))
        self.position, self.probes = parts["position"], parts["probes"]

    def _fit(self, train, seed: int) -> Dict[str, object]:
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        from src import warcraft as wc

        print("fitting the WarCraft probes once (about a minute); cached afterwards", flush=True)
        # The per-tile mean of the network's own costs: what step 11's
        # position-only surrogate (row and column, gradient boosting) estimates.
        position = wc.plan(self.planner, train.images[: self.POSITION_MAPS])[0].mean(axis=0)
        acts = wc.tile_activations(self.planner, train.images[: self.PROBE_MAPS])
        target = train.weights[: self.PROBE_MAPS].reshape(-1)
        probes = {}
        for layer in ("block3", "output"):
            x = acts[layer].reshape(-1, acts[layer].shape[-1])
            probes[layer] = make_pipeline(StandardScaler(), RidgeCV()).fit(x, target)
        return {"position": position, "probes": probes}

    def explain(self, image: np.ndarray, weights: np.ndarray, cost: np.ndarray) -> Dict[str, object]:
        """Position pattern, the ablation that removes it, and the probes' readouts."""
        from scipy.stats import spearmanr

        from src import warcraft as wc

        ablated = np.clip(cost - self.position + self.position.mean(), 1e-3, None)
        paths, hists = wc.search_cost_maps(self.planner, ablated[None])
        scored = wc.score(weights[None], paths, hists, ablated[None])
        acts = wc.tile_activations(self.planner, image[None])
        readout = {layer: probe.predict(acts[layer].reshape(-1, acts[layer].shape[-1])).reshape(cost.shape)
                   for layer, probe in self.probes.items()}
        return {
            "position": self.position,
            "rank_position": float(spearmanr(cost.ravel(), self.position.ravel()).correlation),
            "ablated_cost": float(scored["path_cost"][0]),
            "ablated_route": [tuple(int(v) for v in rc) for rc in np.argwhere(paths[0] > 0)],
            "ablated_path_map": paths[0],
            "readout": readout,
            "probe_r2": {layer: r2_on(weights.ravel(), r.ravel()) for layer, r in readout.items()},
        }


def main() -> int:
    """Print what the demo shows for one clean test map, without the browser."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map", type=int, default=0, help="test map")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from src.data import load_split
    from src.exact_solver import solve_dijkstra
    from src.planner import encode_cost_map, load_pretrained_neural_astar

    config.set_global_seeds(args.seed)
    planner = load_pretrained_neural_astar()
    split = load_split("test")
    inst = split.instance(args.map, seed=args.seed)
    maps, starts, goals = inst.to_tensors()
    with torch.no_grad():
        cost = encode_cost_map(planner, maps, starts, goals)[0, 0].numpy()
    e = MazeExplainer(planner, split, seed=args.seed).explain(inst, inst.map_design, cost, clean=True)
    print(f"map {args.map}: optimum {solve_dijkstra(inst.map_design, inst.start_rc, inst.goal_rc).cost:.0f} moves")
    for group, s in e["surrogates"].items():
        print(f"  surrogate {group:12s} R2 this map {s['r2_this_map']:+.3f}  route {s['moves']:.0f}  expanded {s['expanded']:.0f}")
    for name, label in CORRELATES.items():
        print(f"  r with {label:30s} {e['correlations'][name]:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
