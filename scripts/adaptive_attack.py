"""PLAN.md step 13: does step 12's surviving detector claim hold against an attacker who knows it?

Step 12's one surviving claim: on mazes, for the single flip that most
increases search effort (`k1_eff`), adding the surrogate residual to a
convolutional map-only control raises detection AUC by about +0.08 to +0.11,
on all four models. Every attack it was measured on chose its flip for
*damage alone*. A detector is only evaluated properly against an attacker who
knows it is there and adapts (Carlini & Wagner 2017; Tramer et al. 2020).

The adaptive attacker here is white-box and given every advantage:

1. Step 12's detector is fitted exactly as step 12 fits it, on the first half
   of the maps: clean map vs. its worst efficiency flip, features = the
   convolutional control's score plus the seven residual statistics.
2. On each held-out map the attacker takes the single flips that still do at
   least a fraction `f` of the worst flip's Exp damage (with Opt kept within
   the efficiency cap) and plays the one the fitted detector scores *least*
   altered. `f = 1` is the step-12 attack, so the first row reproduces it.
   A second attacker keeps step 12's flip and adds one more cell anywhere on
   the map, chosen only to hide the first: a camouflage flip.
3. The fixed detector is scored on those flips. The question is whether the
   residual's gain over the control survives, and how much damage the
   attacker gave up to erase it.

The detector is not refitted on the adaptive flips. A defender who retrains
would do better; that is the next round of the game, not this one.

Usage:
    python scripts/adaptive_attack.py --instances 20 --timing-only
    python scripts/adaptive_attack.py
    python scripts/adaptive_attack.py --tag mazes_seed0 --checkpoint results/models/mazes_seed0
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.e4_detector import (  # noqa: E402
    clean_residual_sd,
    make_feature_builder,
    select_alterations,
    train_grid_control,
)
from src import config  # noqa: E402
from src.data import load_split  # noqa: E402
from src.detect import (  # noqa: E402
    bootstrap_auc_gap,
    fit_detector,
    grid_cnn_inputs,
    residual_features,
    score_grid_cnn,
    unpaired_mask,
)
from src.grid import flip_cells  # noqa: E402
from src.grid import is_reachable  # noqa: E402
from src.perturb import (  # noqa: E402
    OPT_CAP_FOR_EFFICIENCY,
    AlterationRequest,
    Baseline,
    candidate_cells,
    evaluate_alterations,
)
from src.planner import load_pretrained_neural_astar  # noqa: E402
from src.surrogate import build_design_matrix, fit_and_score  # noqa: E402

FRACTIONS = (1.0, 0.9, 0.75, 0.5)
MAX_CANDIDATES = 64
MAX_PAIRS = 256


def main() -> int:
    """Fit step 12's detector, attack it adaptively, and report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--instances", type=int, default=None, help="cap, for probes")
    parser.add_argument("--group", default="geometric_h1")
    parser.add_argument("--kind", default="gbm", choices=["ridge", "gbm"])
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--tag", default="")
    parser.add_argument("--grid-cnn-epochs", type=int, default=40)
    parser.add_argument("--no-pairs", action="store_true", help="single-flip attacker only")
    parser.add_argument("--timing-only", action="store_true")
    args = parser.parse_args()

    config.set_global_seeds(args.seed)
    planner = load_pretrained_neural_astar(checkpoint_path=args.checkpoint)
    suffix = f"_{args.tag}" if args.tag else ""
    surrogate = fit_and_score(
        build_design_matrix("train", tag=args.tag), build_design_matrix(args.split, tag=args.tag),
        args.group, kind=args.kind, seed=args.seed,
    )
    builder = make_feature_builder(args.group)
    instances = load_split(args.split).instances(seed=args.seed)
    by_id = {inst.instance_id: inst for inst in instances}
    spread = clean_residual_sd(planner, surrogate, load_split("valid").instances(seed=args.seed), builder)
    started = time.perf_counter()
    cnn, cnn_scores = train_grid_control("train", "mazes", args.seed, args.grid_cnn_epochs)
    print(f"surrogate R² {surrogate.r2:.3f}; convolutional control trained "
          f"({time.perf_counter() - started:.0f} s, held-out random-flip AUC "
          f"{cnn_scores['random_flip_auc_valid']:.3f})")

    manifest = pd.read_parquet(config.RESULTS_DIR / f"manifest{suffix}.parquet")
    manifest = manifest[manifest["split"] == args.split]
    worst = select_alterations(manifest, "k1_eff")
    ids = sorted(worst)
    if args.instances is not None:
        ids = ids[: args.instances]
    first = set(ids[: len(ids) // 2])
    held = [i for i in ids if i not in first]

    def features(instance_id: int, design: np.ndarray) -> np.ndarray:
        """Control score followed by the residual statistics, for one map."""
        instance = by_id[instance_id]
        column = score_grid_cnn(cnn, grid_cnn_inputs([instance], [design]))[0]
        return np.array([column] + residual_features(planner, surrogate, instance, design, builder, spread))

    clean = {i: features(i, by_id[i].map_design) for i in ids}
    train_x = np.array([clean[i] for i in first] + [features(i, flip_cells(by_id[i].map_design, worst[i]))
                                                   for i in first])
    train_y = np.array([0] * len(first) + [1] * len(first))
    combined_head = fit_detector(train_x, train_y)
    control_head = fit_detector(train_x[:, :1], train_y)

    # Candidate alterations on held-out maps, admissible for the efficiency
    # objective and scored once by the fixed detector. Two attacker families:
    # `single` plays any one flip; `pair` keeps step 12's flip and adds one
    # more cell anywhere, whose only job is camouflage. Damage is always
    # measured against step 12's single worst flip on that map.
    exhaustive = manifest[(manifest["method"] == "exhaustive") & (manifest["budget"] == 1)]
    baselines: Dict[int, Baseline] = {}
    rng = np.random.default_rng(args.seed)
    candidates: Dict[str, Dict[int, pd.DataFrame]] = {"single": {}, "pair": {}}
    started = time.perf_counter()
    for position, instance_id in enumerate(held, start=1):
        instance = by_id[instance_id]
        rows = exhaustive[(exhaustive["instance_id"] == instance_id)
                          & (exhaustive["opt_C1O1"] <= OPT_CAP_FOR_EFFICIENCY)].copy()
        rows["damage"] = rows["exp_C1O1"] - rows["exp_C0O0"]
        top = float(rows["damage"].max()) if len(rows) else 0.0
        if top <= 0:
            continue  # no flip raises search effort here: nothing to hide
        families = {"single": rows[rows["damage"] >= min(FRACTIONS) * top].nlargest(MAX_CANDIDATES, "damage")}
        if not args.no_pairs:
            requests = []
            for cell in candidate_cells(instance.map_design, instance.start_rc, instance.goal_rc):
                if cell in worst[instance_id]:
                    continue
                pair = tuple(sorted(worst[instance_id] + [cell]))
                if is_reachable(flip_cells(instance.map_design, pair), instance.start_rc, instance.goal_rc):
                    requests.append(AlterationRequest(instance=instance, flip_cells=pair,
                                                      method="adaptive", objective="efficiency"))
            pairs = pd.DataFrame(evaluate_alterations(planner, requests, baselines=baselines, batch_size=256))
            pairs["damage"] = pairs["exp_C1O1"] - pairs["exp_C0O0"]
            pairs = pairs[(pairs["opt_C1O1"] <= OPT_CAP_FOR_EFFICIENCY)
                          & (pairs["damage"] >= min(FRACTIONS) * top)]
            if len(pairs) > MAX_PAIRS:
                pairs = pairs.iloc[np.sort(rng.choice(len(pairs), MAX_PAIRS, replace=False))]
            families["pair"] = pairs
        for family, frame in families.items():
            if frame.empty:
                continue
            frame = frame.copy()
            cells = [[(int(c[0]), int(c[1])) for c in cell] for cell in frame["flip_cells"]]
            matrix = np.array([features(instance_id, flip_cells(instance.map_design, c)) for c in cells])
            frame["score"] = combined_head.predict_proba(matrix)[:, 1]
            frame["cells"] = cells
            frame["vector"] = list(matrix)
            frame["top_damage"] = top
            candidates[family][instance_id] = frame
        if position % 10 == 0 or position == len(held):
            print(f"  scored candidates on {position}/{len(held)} held-out maps "
                  f"({time.perf_counter() - started:.0f} s)", flush=True)
    dropped = [i for i in held if i not in candidates["single"]]
    print(f"  {len(dropped)} held-out maps have no flip that raises Exp and are left out: {dropped}")
    if args.timing_only:
        return 0

    from sklearn.metrics import roc_auc_score

    results: List[Dict[str, object]] = []
    for family, by_map in candidates.items():
        for fraction in FRACTIONS:
            chosen = {}
            for instance_id, frame in by_map.items():
                pool = frame[frame["damage"] >= fraction * frame["top_damage"] - 1e-9]
                if len(pool):
                    chosen[instance_id] = pool.loc[pool["score"].idxmin()]
            maps = sorted(chosen)
            if len(maps) < 8:
                continue
            held_x = np.array([clean[i] for i in maps] + [chosen[i]["vector"] for i in maps])
            held_y = np.array([0] * len(maps) + [1] * len(maps))
            x = np.vstack([train_x, held_x])
            y = np.concatenate([train_y, held_y])
            ids_rows = list(first) * 2 + maps * 2
            is_train = np.array([True] * len(train_y) + [False] * len(held_y))
            gap = bootstrap_auc_gap(x, x[:, :1], y, is_train, ids_rows, seed=args.seed,
                                    models=(combined_head, control_head))
            keep = unpaired_mask(y, is_train, ids_rows)
            unpaired = bootstrap_auc_gap(x[keep], x[keep][:, :1], y[keep], is_train[keep],
                                         [r for r, k in zip(ids_rows, keep) if k], seed=args.seed,
                                         models=(combined_head, control_head))
            ev = ~is_train
            kept = np.array([chosen[i]["damage"] / chosen[i]["top_damage"] for i in maps])
            results.append({
                "family": family,
                "fraction": fraction,
                "n_held_out": len(maps),
                "control_auc": float(roc_auc_score(y[ev], control_head.predict_proba(x[ev][:, :1])[:, 1])),
                "combined_auc": float(roc_auc_score(y[ev], combined_head.predict_proba(x[ev])[:, 1])),
                "gap": gap,
                "unpaired_gap": unpaired,
                "mean_damage_kept": float(kept.mean()),
                "min_damage_kept": float(kept.min()),
                "frac_maps_not_step12_flip": float(np.mean([chosen[i]["cells"] != worst[i] for i in maps])),
                "mean_candidates": float(np.mean([
                    (by_map[i]["damage"] >= fraction * by_map[i]["top_damage"] - 1e-9).sum() for i in maps])),
            })
            r = results[-1]
            print(f"{family} f={fraction:.2f} (n={len(maps)}): control {r['control_auc']:.3f} | "
                  f"+residual {r['combined_auc']:.3f} | gap {gap['gap']:+.3f} "
                  f"[{gap['lo']:+.3f}, {gap['hi']:+.3f}] | unpaired gap {unpaired['gap']:+.3f} | "
                  f"damage kept {r['mean_damage_kept']:.2f} | {r['mean_candidates']:.0f} candidates/map")

    payload = {"split": args.split, "seed": args.seed, "model_tag": args.tag or "pretrained",
               "surrogate_r2": surrogate.r2, "grid_cnn_control": cnn_scores,
               "n_train_maps": len(first), "max_candidates": MAX_CANDIDATES, "results": results}
    out = config.RESULTS_DIR / f"e4_adaptive{suffix}.json"
    out.write_text(json.dumps(payload, indent=2, default=float) + "\n")
    lines = [
        f"# Step 13 — adaptive attack on step 12's detector ({args.tag or 'pretrained'})",
        "",
        f"Detector: convolutional map-only control + residual statistics, fitted on {len(first)} maps "
        "(clean vs. worst efficiency flip), never refitted. On each of the "
        f"{len(held)} held-out maps the attacker plays the single flip the detector scores least "
        "altered among those keeping at least a fraction `f` of the worst flip's Exp damage. "
        "`f = 1` is step 12's attack.",
        "",
        "`single`: any one flip. `pair`: step 12's flip plus one more cell anywhere, chosen only to "
        "hide it. Damage kept is the Exp increase over the clean map, divided by that of step 12's "
        "flip. Held-out maps where no flip raises Exp are left out (nothing to hide).",
        "",
        "| attacker | f | maps | control AUC | control + residual | gap, 95 % CI | unpaired gap, 95 % CI "
        "| damage kept (mean / min) | not step 12's alteration | candidates per map |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        g, u = r["gap"], r["unpaired_gap"]
        lines.append(
            f"| {r['family']} | {r['fraction']:.2f} | {r['n_held_out']} | {r['control_auc']:.3f} | {r['combined_auc']:.3f} | "
            f"{g['gap']:+.3f} [{g['lo']:+.3f}, {g['hi']:+.3f}] | {u['gap']:+.3f} [{u['lo']:+.3f}, "
            f"{u['hi']:+.3f}] | {r['mean_damage_kept']:.2f} / {r['min_damage_kept']:.2f} | "
            f"{r['frac_maps_not_step12_flip']:.0%} | {r['mean_candidates']:.1f} |"
        )
    (config.RESULTS_DIR / f"e4_adaptive{suffix}.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out.name} and e4_adaptive{suffix}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
