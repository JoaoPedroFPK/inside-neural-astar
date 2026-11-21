"""PLAN.md step 11: the project's questions on WarCraft 12x12 terrain images.

One model per call (``--tag warcraft_seed0`` reads
``results/models/warcraft_seed0``); outputs go to ``results/warcraft/<tag>/``.
Stages, each cached so the report never recomputes:

* ``eval`` — baseline Opt / Exp on the test split, and how the learned cost
  map relates to the true terrain cost (step-0 analogue).
* ``surrogate`` — ridge fits from interpretable tile features to the cost map,
  and a sufficiency test: plan on the surrogate's map (steps 4-5).
* ``pilot`` — random tile transplants at k = 1, 2, 4, 8 (step 3).
* ``sweep`` — every single-tile transplant on the first ``--instances`` test
  maps (step 6), with the cost-ranking / search-rule decomposition that
  replaces the 2x2 swap (step 7); see the step document for why the swap has
  no obstacle half here.
* ``linf`` — pixel perturbations bounded in L-infinity norm, which leave the
  true costs unchanged: random noise and a projected-gradient attack.

Usage:
    python scripts/warcraft_study.py --tag warcraft_seed0 --stage sweep --instances 3
    python scripts/warcraft_study.py --tag warcraft_seed0 --stage all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src import warcraft as wc  # noqa: E402
from src.surrogate import make_model, within_instance_r2  # noqa: E402

TOL = wc.OPT_TOL


def out_dir(tag: str) -> Path:
    path = config.RESULTS_DIR / "warcraft" / tag
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=float) + "\n")
    print(f"wrote {path.relative_to(config.REPO_ROOT)}")


def summary_of(s: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Aggregate a `wc.score` result."""
    return {
        "n": int(len(s["opt"])),
        "opt_mean": float(s["opt"].mean()),
        "opt_binary": float((s["opt"] <= 1 + TOL).mean()),
        "exp_mean": float(s["exp"].mean()),
        "exp_median": float(np.median(s["exp"])),
        "opt_exact_mean": float(s["opt_exact"].mean()),
        "opt_exact_binary": float((s["opt_exact"] <= 1 + TOL).mean()),
        "valid": float(s["valid"].mean()),
    }


# --------------------------------------------------------------------------- #
# eval
# --------------------------------------------------------------------------- #

def stage_eval(planner, tag: str) -> None:
    from scipy.stats import spearmanr

    test = wc.load_warcraft_split("test")
    costs, paths, hists = wc.plan(planner, test.images)
    s = wc.score(test.weights, paths, hists, costs)
    np.savez_compressed(out_dir(tag) / "test_outputs.npz", cost_maps=costs, paths=paths,
                        histories=hists, opt=s["opt"], exp=s["exp"], opt_exact=s["opt_exact"])

    # The classical control's own path quality, for reference: the same
    # search rule fed the true weights.
    classical = np.empty(len(test))
    for i in range(len(test)):
        sol = wc.solve_astar(wc._FREE, wc.START_RC, wc.GOAL_RC, cost_map=test.weights[i])
        classical[i] = (sol.cost + test.weights[i][wc.GOAL_RC]) / s["optimum"][i]

    per_class = {
        str(c): {
            "mean": float(costs[np.isclose(test.weights, c)].mean()),
            "std": float(costs[np.isclose(test.weights, c)].std()),
        }
        for c in wc.TERRAIN_COSTS
    }
    rho = [spearmanr(costs[i].ravel(), test.weights[i].ravel())[0] for i in range(len(test))]
    on_path = test.paths > 0.5
    write_json(out_dir(tag) / "eval.json", {
        "tag": tag,
        "planner": summary_of(s),
        "classical_true_weights": {
            "opt_mean": float(classical.mean()),
            "opt_binary": float((classical <= 1 + TOL).mean()),
        },
        "cost_by_terrain": per_class,
        "spearman_cost_vs_weight_mean": float(np.nanmean(rho)),
        "cost_on_reference_path_mean": float(costs[on_path].mean()),
        "cost_off_reference_path_mean": float(costs[~on_path].mean()),
        "learned_scale_const": float(planner.encoder.const.item()),
    })


# --------------------------------------------------------------------------- #
# surrogate
# --------------------------------------------------------------------------- #

def stage_surrogate(planner, tag: str, n_train: int) -> None:
    from sklearn.metrics import r2_score

    train = wc.load_warcraft_split("train")
    test = wc.load_warcraft_split("test")
    y_train = wc.plan(planner, train.images[:n_train])[0].reshape(-1)
    y_test = np.load(out_dir(tag) / "test_outputs.npz")["cost_maps"]
    columns = wc.FEATURE_GROUPS["all"]
    X_train = wc.design_matrix(train.weights[:n_train], columns)
    X_test = wc.design_matrix(test.weights, columns)
    instance = np.repeat(np.arange(len(test)), wc.GRID * wc.GRID)

    fits: List[Dict[str, object]] = []
    predictions = {}
    for group, names in wc.FEATURE_GROUPS.items():
        cols = [columns.index(n) for n in names]
        for kind in ("ridge", "gbm") if group in ("terrain", "path", "all") else ("ridge",):
            model = make_model(kind).fit(X_train[:, cols], y_train)
            pred = model.predict(X_test[:, cols])
            predictions[(group, kind)] = pred
            fit = {
                "group": group, "kind": kind, "n_features": len(cols),
                "r2": float(r2_score(y_test.reshape(-1), pred)),
                "r2_within": within_instance_r2(y_test.reshape(-1), pred, instance),
            }
            if kind == "ridge":
                fit["coefficients"] = dict(zip(names, map(float, model.named_steps["model"].coef_)))
            fits.append(fit)
            print(f"  {group:18s} {kind:5s} R2={fit['r2']:.3f}")

    # Sufficiency: does the surrogate's map, searched by the same module, plan?
    sufficiency = {}
    for key in (("terrain", "ridge"), ("terrain", "gbm"), ("path", "gbm"), ("all", "ridge"), ("all", "gbm")):
        maps = np.clip(predictions[key].reshape(-1, wc.GRID, wc.GRID), 1e-3, None)
        paths, hists = wc.search_cost_maps(planner, maps)
        sufficiency["/".join(key)] = summary_of(wc.score(test.weights, paths, hists, maps))
    # The ideal cost estimator: the true weights themselves, same search.
    paths, hists = wc.search_cost_maps(planner, test.weights)
    sufficiency["true_weights"] = summary_of(wc.score(test.weights, paths, hists, test.weights))
    write_json(out_dir(tag) / "surrogate.json", {
        "tag": tag, "n_train_maps": n_train, "fits": fits, "sufficiency": sufficiency,
    })


# --------------------------------------------------------------------------- #
# pilot and sweep
# --------------------------------------------------------------------------- #

def stage_pilot(planner, tag: str, draws: int, seed: int) -> None:
    test = wc.load_warcraft_split("test")
    train = wc.load_warcraft_split("train")
    pool = wc.donor_pool(train.weights)
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(len(test)):
        alts = [wc.random_alterations(test.weights[i], k, pool, rng)
                for k in (1, 2, 4, 8) for _ in range(draws)]
        base, results = wc.evaluate_alterations(planner, test.images[i], test.weights[i], alts, train.images)
        for r in results:
            rows.append({"instance_id": i, "opt_C0": base["opt"], "exp_C0": base["exp"], **r})
    df = pd.DataFrame(rows)
    df.to_parquet(out_dir(tag) / "pilot.parquet")
    table = []
    for k, g in df.groupby("budget"):
        table.append({
            "budget": int(k), "n": int(len(g)),
            "frac_suboptimal": float((g.opt_C1 > 1 + TOL).mean()),
            "frac_worsened": float((g.opt_C1 > g.opt_C0 + TOL).mean()),
            "frac_exp_gt_1_5": float((g.exp_C1 > 1.5).mean()),
            "frac_exp_up_1_5x": float((g.exp_C1 > 1.5 * g.exp_C0).mean()),
            "opt_mean": float(g.opt_C1.mean()),
            "exp_mean": float(g.exp_C1.mean()),
            "frac_path_changed": float(g.path_changed.mean()),
        })
    write_json(out_dir(tag) / "pilot.json", {"tag": tag, "draws": draws, "seed": seed, "by_budget": table})


def stage_sweep(planner, tag: str, n_instances: int, seed: int) -> None:
    test = wc.load_warcraft_split("test")
    train = wc.load_warcraft_split("train")
    pool = wc.donor_pool(train.weights)
    rng = np.random.default_rng(seed)
    rows = []
    started = time.perf_counter()
    for i in range(n_instances):
        alts = wc.k1_alterations(test.weights[i], pool, rng)
        base, results = wc.evaluate_alterations(planner, test.images[i], test.weights[i], alts, train.images)
        features = wc.tile_features(test.weights[i])
        for r in results:
            rc = tuple(r["flip_cells"][0])
            r["on_optimal_path"] = bool(features["on_optimal_path"][rc])
            r["cheb_to_optimal_path"] = float(features["cheb_to_optimal_path"][rc])
            r["on_planner_path"] = bool(base["path"][rc] > 0.5)
            rows.append({
                "instance_id": i, "split": "test", "seed": seed, "model": tag,
                "start_rc": wc.START_RC, "goal_rc": wc.GOAL_RC,
                "method": "exhaustive", "objective": "none", "reachable": True,
                "opt_C0": base["opt"], "exp_C0": base["exp"], "opt_exact_C0": base["opt_exact"],
                **r,
            })
        if i < 3 or (i + 1) % 50 == 0:
            per_map = (time.perf_counter() - started) / (i + 1)
            print(f"  map {i + 1}/{n_instances}: {per_map:.1f} s/map, "
                  f"{per_map * n_instances / 60:.0f} min total", flush=True)
    df = pd.DataFrame(rows)
    df["flip_cells"] = df["flip_cells"].map(lambda cells: [list(c) for c in cells])
    df.to_parquet(out_dir(tag) / "manifest.parquet")
    write_json(out_dir(tag) / "sweep.json", summarize_sweep(df, tag))


def summarize_sweep(df: pd.DataFrame, tag: str) -> Dict[str, object]:
    """Fragility, where the damage is, and what carries it.

    *Damage* on a map the planner solved optimally means the altered map is not
    solved optimally (``opt_C1 > 1``). On maps it already solved
    sub-optimally, *worsened* means ``opt_C1 > opt_C0``.

    The decomposition (replacing the cost/obstacle swap, whose obstacle half is
    empty here) splits each damaging edit by whether an exact search on the
    altered cost map also misses the optimum:

    * **cost-ranking** — it does: the encoder's costs, searched perfectly,
      already prefer a worse path;
    * **search-rule** — it does not: the costs rank an optimal path first, and
      the planner's own search rule (heuristic, ``g_ratio``) loses it.
    """
    base_opt = df.groupby("instance_id").opt_C0.first() <= 1 + TOL
    opt_maps = base_opt[base_opt].index
    d = df[df.instance_id.isin(opt_maps)].copy()
    d["damaging"] = d.opt_C1 > 1 + TOL
    d["exp_damaging"] = d.exp_C1 > 1.5 * d.exp_C0
    per_map = d.groupby("instance_id").damaging.agg(["any", "mean"])

    dmg = d[d.damaging]
    decomposition = {
        "n_damaging": int(len(dmg)),
        "cost_ranking": float((dmg.opt_exact_C1 > 1 + TOL).mean()) if len(dmg) else float("nan"),
        "search_rule": float((dmg.opt_exact_C1 <= 1 + TOL).mean()) if len(dmg) else float("nan"),
        "ignoring_change_was_optimal": float((dmg.opt_C0_on_W1 <= 1 + TOL).mean()) if len(dmg) else float("nan"),
        "opt_C1_mean_damaging": float(dmg.opt_C1.mean()) if len(dmg) else float("nan"),
    }
    by_direction = []
    d["direction"] = np.where(
        d.new_weight.map(lambda w: w[0]) > d.original_weight.map(lambda w: w[0]), "costlier", "cheaper"
    )
    for name, g in d.groupby("direction"):
        by_direction.append({
            "direction": name, "n": int(len(g)),
            "frac_damaging": float(g.damaging.mean()),
            "frac_needed_response": float((g.opt_C0_on_W1 > 1 + TOL).mean()),
        })
    by_location = []
    d["location"] = np.select(
        [d.on_planner_path, d.on_optimal_path, d.cheb_to_optimal_path <= 1],
        ["on planner path", "on another optimal path", "next to an optimal path"],
        "away from optimal paths",
    )
    for name, g in d.groupby("location"):
        by_location.append({
            "location": name, "n": int(len(g)),
            "frac_damaging": float(g.damaging.mean()),
            "frac_damaging_costlier": float(g[g.direction == "costlier"].damaging.mean()),
            "frac_damaging_cheaper": float(g[g.direction == "cheaper"].damaging.mean()),
        })
    worsened_all = df[df.opt_C1 > df.opt_C0 + TOL]
    return {
        "tag": tag,
        "n_maps": int(df.instance_id.nunique()),
        "n_alterations": int(len(df)),
        "n_baseline_optimal_maps": int(len(opt_maps)),
        "frac_optimal_maps_with_damaging_tile": float(per_map["any"].mean()),
        "median_frac_damaging_edits_per_map": float(per_map["mean"].median()),
        "frac_damaging_edits": float(d.damaging.mean()),
        "frac_exp_damaging_edits": float(d.exp_damaging.mean()),
        "worst_opt": float(df.opt_C1.max()),
        "frac_worsened_all_maps": float(len(worsened_all) / len(df)),
        "frac_needed_response": float((df.opt_C0_on_W1 > 1 + TOL).mean()),
        "frac_path_changed": float(df.path_changed.mean()),
        "max_dcost_outside_receptive_field": float(df.dcost_outside_receptive_field.max()),
        "dcost_frac_under_flips_mean": float(df.dcost_frac_under_flips.mean()),
        "spearman_dcost_vs_opt_damage": float(
            d[["dcost_l1"]].assign(dmg=d.opt_C1 - 1).corr(method="spearman").iloc[0, 1]
        ),
        "decomposition": decomposition,
        "by_direction": by_direction,
        "by_location": by_location,
    }


# --------------------------------------------------------------------------- #
# L-infinity pixel perturbations
# --------------------------------------------------------------------------- #

def pgd(planner, images: np.ndarray, targets: np.ndarray, eps: float, steps: int) -> np.ndarray:
    """Projected-gradient ascent on upstream's own training loss.

    Maximises the L1 distance between the differentiable search's expanded
    set and the true optimal path (the loss the model was trained to
    minimise), within ``||delta||_inf <= eps`` and the valid pixel range.
    """
    x0 = torch.from_numpy(images.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
    _, starts, goals = wc.to_tensors(images[:1])
    starts, goals = starts.expand(len(images), -1, -1, -1), goals.expand(len(images), -1, -1, -1)
    target = torch.from_numpy(targets.astype(np.float32))[:, None]
    delta = torch.zeros_like(x0)
    alpha = eps / 4
    for _ in range(steps):
        delta.requires_grad_(True)
        out = planner(torch.clamp(x0 + delta, 0, 1), starts, goals)
        loss = (out.histories - target).abs().mean()
        (grad,) = torch.autograd.grad(loss, delta)
        with torch.no_grad():
            delta = torch.clamp(delta + alpha * grad.sign(), -eps, eps)
            delta = torch.clamp(x0 + delta, 0, 1) - x0
    return (torch.clamp(x0 + delta, 0, 1)).permute(0, 2, 3, 1).numpy()


def stage_linf(planner, tag: str, n_instances: int, seed: int, steps: int) -> None:
    test = wc.load_warcraft_split("test")
    images, weights, ref = test.images[:n_instances], test.weights[:n_instances], test.paths[:n_instances]
    rng = np.random.default_rng(seed)
    c0, p0, h0 = wc.plan(planner, images)
    s0 = wc.score(weights, p0, h0, c0)
    rows = []
    for eps_255 in (1, 2, 4, 8, 16):
        eps = eps_255 / 255.0
        noisy = np.clip(images / 255.0 + rng.choice([-eps, eps], size=images.shape), 0, 1)
        attacked = np.concatenate([
            pgd(planner, images[b : b + 50], ref[b : b + 50], eps, steps)
            for b in range(0, n_instances, 50)
        ])
        for method, x in (("random_sign", noisy), ("pgd", attacked)):
            c1, p1, h1 = wc.plan(planner, x)
            s1 = wc.score(weights, p1, h1, c1)
            optimal0 = s0["opt"] <= 1 + TOL
            rows.append({
                "eps_255": eps_255, "method": method, "n": int(n_instances),
                "opt_mean": float(s1["opt"].mean()),
                "opt_binary": float((s1["opt"] <= 1 + TOL).mean()),
                "frac_optimal_maps_broken": float((s1["opt"][optimal0] > 1 + TOL).mean()),
                "frac_worsened": float((s1["opt"] > s0["opt"] + TOL).mean()),
                "exp_mean": float(s1["exp"].mean()),
                "frac_path_changed": float(np.any(p1 != p0, axis=(1, 2)).mean()),
                "opt_exact_binary": float((s1["opt_exact"] <= 1 + TOL).mean()),
                "dcost_l1_mean": float(np.abs(c1 - c0).sum(axis=(1, 2)).mean()),
            })
            print(f"  eps={eps_255}/255 {method:11s} opt_binary={rows[-1]['opt_binary']:.3f} "
                  f"broken={rows[-1]['frac_optimal_maps_broken']:.3f}", flush=True)
    write_json(out_dir(tag) / "linf.json", {
        "tag": tag, "n_instances": n_instances, "pgd_steps": steps,
        "clean": summary_of(s0), "rows": rows,
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True, help="model directory name under results/models/")
    parser.add_argument("--stage", default="all",
                        choices=["eval", "surrogate", "pilot", "sweep", "linf", "all"])
    parser.add_argument("--instances", type=int, default=300, help="test maps for sweep / linf")
    parser.add_argument("--surrogate-train-maps", type=int, default=2000)
    parser.add_argument("--draws", type=int, default=3)
    parser.add_argument("--pgd-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config.set_global_seeds(args.seed)
    planner = wc.load_warcraft_planner(config.RESULTS_DIR / "models" / args.tag)
    stages = ["eval", "surrogate", "pilot", "sweep", "linf"] if args.stage == "all" else [args.stage]
    for stage in stages:
        print(f"==> {args.tag}: {stage}", flush=True)
        started = time.perf_counter()
        if stage == "eval":
            stage_eval(planner, args.tag)
        elif stage == "surrogate":
            stage_surrogate(planner, args.tag, args.surrogate_train_maps)
        elif stage == "pilot":
            stage_pilot(planner, args.tag, args.draws, args.seed)
        elif stage == "sweep":
            stage_sweep(planner, args.tag, args.instances, args.seed)
        elif stage == "linf":
            stage_linf(planner, args.tag, args.instances, args.seed, args.pgd_steps)
        print(f"    {stage} took {(time.perf_counter() - started) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
