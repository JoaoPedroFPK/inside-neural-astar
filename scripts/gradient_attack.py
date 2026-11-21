"""PLAN.md step 13: a gradient-guided flip attack that needs no exhaustive sweep.

The greedy attack of step 9 draws its candidates from the exhaustive k=1
sweep: before choosing anything it has already tried every single cell. That
is an oracle no real attacker has, and every detector result of step 12 was
measured against attacks built on it. This script replaces the oracle with
the planner's own gradient (`src.perturb.flip_sensitivity`): each round, rank
every cell by the first-order size of its effect, try the top `pool` cells
exactly, keep the best — the same search, a different source of candidates.

Two measurements:

1. **Does the gradient rank cells at all?** Checked against the exhaustive
   sweep, which records the true effect of every single flip. `PLAN.md` asks
   for this test before any gradient search is built.
2. **How much damage does the search do**, against two references at equal
   cost: the same search with a *random* pool (the control: what the gradient
   adds over no information), and step 9's sweep-seeded greedy (the ceiling:
   what perfect single-cell knowledge buys).

Maps 0-9 were used to choose the differentiable stand-in for each objective
(three were compared; their differences were small). Every comparison is
also reported on maps 10-99 alone.

Usage:
    python scripts/gradient_attack.py --instances 3 --timing-only
    python scripts/gradient_attack.py
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
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.data import load_split  # noqa: E402
from src.perturb import (  # noqa: E402
    Baseline,
    flip_sensitivity,
    gradient_pool,
    greedy_flip_search,
    random_pool,
)
from src.planner import load_pretrained_neural_astar  # noqa: E402

BUDGETS = (1, 2, 4, 8)
METRIC = {"optimality": "opt_C1O1", "efficiency": "exp_C1O1"}
TUNING_MAPS = 10
TOP = 16
TOL = 1e-6


def ranking_test(planner, split, exhaustive: pd.DataFrame, ids: List[int], seed: int) -> pd.DataFrame:
    """Score the gradient's ranking of single flips against their true effect.

    For each map and objective: Spearman correlation between the score and the
    measured metric over every admissible cell (signed and absolute), the
    share of the top-16 cells that are damaging against the share overall,
    and the best damage among the top 16 against the expected best of 16
    random cells. "Damaging" is: Opt rises above the map's minimum, for
    optimality; Exp in the map's top decile, for efficiency.
    """
    rows = []
    for instance_id in ids:
        instance = split.instance(instance_id, seed=seed)
        group = exhaustive[exhaustive["instance_id"] == instance_id]
        cells = np.array([cell[0] for cell in group["flip_cells"]])
        rng = np.random.default_rng([seed, instance_id])
        for objective, column in METRIC.items():
            y = group[column].to_numpy(dtype=float)
            damaging = y > y.min() + TOL if objective == "optimality" else y >= np.quantile(y, 0.9)
            score = flip_sensitivity(planner, instance, objective)
            pred = score[cells[:, 0], cells[:, 1]]
            top = np.argsort(-pred, kind="stable")[:TOP]
            random_best = np.mean([y[rng.choice(len(y), TOP, replace=False)].max() for _ in range(200)])
            rows.append(
                {
                    "instance_id": instance_id,
                    "objective": objective,
                    "spearman_abs": spearmanr(pred, y).correlation,
                    "hit_rate_top": float(damaging[top].mean()),
                    "hit_rate_all": float(damaging.mean()),
                    "best_top_over_max": float(y[top].max() / y.max()),
                    "best_random_over_max": float(random_best / y.max()),
                    "top_contains_worst": bool(y[top].max() >= y.max() - TOL),
                }
            )
    return pd.DataFrame(rows)


def damage_table(frame: pd.DataFrame) -> List[Dict[str, object]]:
    """Per objective, source and budget: mean damage and share of maps broken."""
    out = []
    for (objective, source, budget), group in frame.groupby(["objective", "candidate_source", "budget"]):
        if budget not in BUDGETS:
            continue
        entry = {"objective": objective, "candidate_source": source, "budget": int(budget),
                 "n": int(len(group))}
        if objective == "optimality":
            optimal = group[group["opt_C0O0"] <= 1 + TOL]
            entry["mean_opt"] = float(group["opt_C1O1"].mean())
            entry["frac_optimal_broken"] = float((optimal["opt_C1O1"] > 1 + TOL).mean())
            entry["n_optimal"] = int(len(optimal))
        else:
            entry["mean_exp"] = float(group["exp_C1O1"].mean())
            entry["mean_exp_gain"] = float((group["exp_C1O1"] / group["exp_C0O0"]).mean())
        out.append(entry)
    return out


def paired_gap(frame: pd.DataFrame, objective: str, budget: int, a: str, b: str, seed: int) -> Dict[str, float]:
    """Mean per-map difference a - b in the objective's metric, 95 % bootstrap CI."""
    column = METRIC[objective]
    sub = frame[(frame["objective"] == objective) & (frame["budget"] == budget)]
    wide = sub.pivot_table(index="instance_id", columns="candidate_source", values=column)
    wide = wide.dropna(subset=[a, b])
    diff = (wide[a] - wide[b]).to_numpy()
    if not len(diff):
        return {"gap": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    boots = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
    return {"gap": float(diff.mean()), "lo": float(np.quantile(boots, 0.025)),
            "hi": float(np.quantile(boots, 0.975)), "n": int(len(diff))}


def render_markdown(report: Dict[str, object]) -> str:
    """The step-13 report."""
    meta = report["meta"]
    lines = [
        "# Step 13 — gradient-guided flip attack",
        "",
        f"Split `{meta['split']}`, seed {meta['seed']}, {meta['n_instances']} maps, pool "
        f"{meta['pool']} cells per round, budgets up to {meta['budget']}. Run time "
        f"{meta['minutes']:.1f} min.",
        "",
        "## 1. Does the gradient rank single flips?",
        "",
        "Against the exhaustive sweep, which knows the true effect of every single flip. "
        f"`top {TOP}` is the {TOP} cells with the largest first-order score.",
        "",
        "| objective | maps | Spearman (abs score) | damaging share, top 16 | damaging share, all cells "
        "| best of top 16 / worst cell | best of 16 random / worst cell | top 16 holds the worst cell |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label, rows in (("all", report["ranking"]["all"]), ("held out", report["ranking"]["held_out"])):
        for objective, r in rows.items():
            lines.append(
                f"| {objective} ({label}) | {r['n']} | {r['spearman_abs']:.3f} | {r['hit_rate_top']:.3f} | "
                f"{r['hit_rate_all']:.3f} | {r['best_top_over_max']:.3f} | "
                f"{r['best_random_over_max']:.3f} | {r['top_contains_worst']:.2f} |"
            )
    lines += [
        "",
        "## 2. Damage at each budget",
        "",
        "`gradient`: pool re-ranked each round by the gradient. `random`: the same search with a "
        "random pool of the same size. `k1_sweep`: step 9's greedy, pool read off the exhaustive "
        "sweep (an attacker who has already tried every cell).",
        "",
        "| objective | budget | source | maps | mean Opt | baseline-optimal maps broken | mean Exp | mean Exp / baseline |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in report["damage"]:
        lines.append(
            f"| {r['objective']} | {r['budget']} | `{r['candidate_source']}` | {r['n']} | "
            + (f"{r['mean_opt']:.3f} | {r['frac_optimal_broken']:.3f} ({r['n_optimal']}) | – | – |"
               if r["objective"] == "optimality"
               else f"– | – | {r['mean_exp']:.3f} | {r['mean_exp_gain']:.3f} |")
        )
    lines += [
        "",
        "## 3. Paired gaps (per-map mean difference in Opt or Exp, 95 % bootstrap CI)",
        "",
        "| objective | budget | comparison | maps | gap | 95 % CI | held-out gap | held-out CI |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in report["gaps"]:
        lines.append(
            f"| {r['objective']} | {r['budget']} | {r['comparison']} | {r['all']['n']} | "
            f"{r['all']['gap']:+.3f} | [{r['all']['lo']:+.3f}, {r['all']['hi']:+.3f}] | "
            f"{r['held_out']['gap']:+.3f} | [{r['held_out']['lo']:+.3f}, {r['held_out']['hi']:+.3f}] |"
        )
    lines += [
        "",
        f"Planner evaluations per map and objective: gradient and random {meta['evaluations_per_search']:,} "
        f"(plus {meta['budget']} gradient passes for `gradient`); `k1_sweep` the same plus the "
        f"{meta['sweep_evaluations_per_map']:,} of the exhaustive sweep.",
    ]
    return "\n".join(lines)


def main() -> int:
    """Run the ranking test and both searches; write the manifest rows and the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=int, default=max(BUDGETS))
    parser.add_argument("--pool", type=int, default=64, help="candidates per round")
    parser.add_argument("--instances", type=int, default=None)
    parser.add_argument("--objectives", nargs="+", default=list(METRIC))
    parser.add_argument("--timing-only", action="store_true")
    parser.add_argument("--manifest", type=Path, default=config.RESULTS_DIR / "manifest.parquet")
    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"{args.manifest} not found; run scripts/sweep_k1.py first", file=sys.stderr)
        return 1
    split_name = config.canonical_split(args.split)
    config.set_global_seeds(args.seed)
    split = load_split(split_name)
    planner = load_pretrained_neural_astar()
    manifest = pd.read_parquet(args.manifest)
    exhaustive = manifest[(manifest["method"] == "exhaustive") & (manifest["budget"] == 1)]
    ids = sorted(int(i) for i in exhaustive["instance_id"].unique())
    if args.instances is not None:
        ids = ids[: args.instances]

    started = time.perf_counter()
    ranking = ranking_test(planner, split, exhaustive, ids, args.seed)
    print(f"ranking test: {time.perf_counter() - started:.1f} s")
    print(ranking.groupby("objective").mean(numeric_only=True).drop(columns="instance_id").round(3))

    rows: List[Dict[str, object]] = []
    baselines: Dict[int, Baseline] = {}
    started = time.perf_counter()
    for objective in args.objectives:
        for position, instance_id in enumerate(ids, start=1):
            instance = split.instance(instance_id, seed=args.seed)
            for source, method, pool in (
                ("gradient", "gradient", gradient_pool(planner, instance, objective, args.pool)),
                ("random", "greedy", random_pool(instance, args.pool, args.seed)),
            ):
                rows.extend(
                    greedy_flip_search(
                        planner, instance, budget=args.budget, objective=objective,
                        candidate_pool=pool, baselines=baselines, method=method,
                        extra={"candidate_source": source},
                    )
                )
            if position % 10 == 0 or position == len(ids):
                print(f"  {objective}: {position}/{len(ids)} maps | "
                      f"{(time.perf_counter() - started) / 60:.1f} min", flush=True)
    seconds = time.perf_counter() - started
    print(f"{len(rows):,} rows in {seconds / 60:.1f} min")
    if args.timing_only:
        scale = len(exhaustive["instance_id"].unique()) / max(len(ids), 1)
        print(f"extrapolation to all maps: {seconds * scale / 60:.0f} min")
        return 0

    new = pd.DataFrame(rows)
    new["flip_cells"] = new["flip_cells"].apply(lambda cells: [list(rc) for rc in cells])
    for column in ("start_rc", "goal_rc"):
        new[column] = new[column].apply(list)
    # Idempotent: drop rows an earlier run of this script wrote, and label the
    # step-9 greedy rows with where their pool came from.
    if "candidate_source" not in manifest.columns:
        manifest["candidate_source"] = None
    greedy = manifest["method"] == "greedy"
    manifest.loc[greedy & manifest["candidate_source"].isna(), "candidate_source"] = "k1_sweep"
    stale = (manifest["method"] == "gradient") | (manifest["candidate_source"] == "random")
    combined = pd.concat([manifest[~stale], new], ignore_index=True)
    combined.to_parquet(args.manifest, index=False)

    searched = combined[combined["candidate_source"].isin(["gradient", "random", "k1_sweep"])
                        & combined["instance_id"].isin(ids)
                        & combined["objective"].isin(args.objectives)]
    held = searched[searched["instance_id"] >= TUNING_MAPS]

    def summarise(frame: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        return {
            objective: {"n": int(len(group)), **group.drop(columns=["instance_id", "objective"]).mean().to_dict()}
            for objective, group in frame.groupby("objective")
        }

    gaps = []
    for objective in args.objectives:
        for budget in BUDGETS:
            for a, b in (("gradient", "random"), ("gradient", "k1_sweep")):
                gaps.append({
                    "objective": objective, "budget": budget, "comparison": f"{a} − {b}",
                    "all": paired_gap(searched, objective, budget, a, b, args.seed),
                    "held_out": paired_gap(held, objective, budget, a, b, args.seed),
                })
    report = {
        "meta": {
            "split": split_name, "seed": args.seed, "pool": args.pool, "budget": args.budget,
            "n_instances": len(ids), "tuning_maps": list(range(TUNING_MAPS)),
            "minutes": seconds / 60,
            "evaluations_per_search": args.pool * args.budget,
            "sweep_evaluations_per_map": int(len(exhaustive) / max(len(exhaustive["instance_id"].unique()), 1)),
        },
        "ranking": {"all": summarise(ranking),
                    "held_out": summarise(ranking[ranking["instance_id"] >= TUNING_MAPS])},
        "damage": damage_table(searched),
        "damage_held_out": damage_table(held),
        "gaps": gaps,
    }
    (config.RESULTS_DIR / "e2_gradient.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    (config.RESULTS_DIR / "e2_gradient.md").write_text(render_markdown(report) + "\n")
    print(f"wrote e2_gradient.md and e2_gradient.json; manifest now {len(combined):,} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
