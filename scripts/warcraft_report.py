"""PLAN.md step 11: gather the per-model WarCraft results into one report.

Reads ``results/warcraft/<tag>/{eval,surrogate,pilot,sweep,linf}.json`` for
every tag given and writes ``results/warcraft.{md,json}``. Every headline is
quoted as a range across models, never as one model's value.

Usage:
    python scripts/warcraft_report.py --tags warcraft_seed0 warcraft_seed1 warcraft_seed2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402


def load(tag: str, name: str) -> Dict[str, object]:
    return json.loads((config.RESULTS_DIR / "warcraft" / tag / f"{name}.json").read_text())


def rng(values: List[float], fmt: str = "{:.3f}") -> str:
    lo, hi = min(values), max(values)
    return fmt.format(lo) if fmt.format(lo) == fmt.format(hi) else f"{fmt.format(lo)}–{fmt.format(hi)}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tags", nargs="+", required=True)
    args = parser.parse_args()
    tags = args.tags
    data = {t: {n: load(t, n) for n in ("eval", "surrogate", "pilot", "sweep", "linf")} for t in tags}
    (config.RESULTS_DIR / "warcraft.json").write_text(json.dumps(data, indent=2, default=float) + "\n")

    def col(stage: str, *keys):
        out = []
        for t in tags:
            v = data[t][stage]
            for k in keys:
                v = v[k]
            out.append(v)
        return out

    L = ["# Step 11 — WarCraft 12×12", "",
         f"Models: {', '.join(f'`{t}`' for t in tags)}. Ranges are min–max across models.", "",
         "## Baseline (test split, 1 000 maps)", "",
         "| quantity | " + " | ".join(tags) + " | range |", "|---|" + "---|" * (len(tags) + 1)]
    rows = [
        ("Opt binary (planner)", ("planner", "opt_binary")),
        ("Opt mean ratio", ("planner", "opt_mean")),
        ("Exp mean ratio", ("planner", "exp_mean")),
        ("Opt binary, exact search on learned costs", ("planner", "opt_exact_binary")),
        ("Opt binary, classical A* on true weights", ("classical_true_weights", "opt_binary")),
        ("Spearman(cost, true weight), per-map mean", ("spearman_cost_vs_weight_mean",)),
        ("learned scale `const`", ("learned_scale_const",)),
    ]
    for label, keys in rows:
        vals = col("eval", *keys)
        L.append(f"| {label} | " + " | ".join(f"{v:.3f}" for v in vals) + f" | {rng(vals)} |")
    L += ["", "Mean learned cost per terrain class:", "",
          "| true cost | " + " | ".join(tags) + " |", "|---|" + "---|" * len(tags)]
    for c in ("0.8", "1.2", "5.3", "7.7", "9.2"):
        L.append(f"| {c} | " + " | ".join(
            f"{data[t]['eval']['cost_by_terrain'][c]['mean']:.2f} ± {data[t]['eval']['cost_by_terrain'][c]['std']:.2f}"
            for t in tags) + " |")

    L += ["", "## Surrogate fits (R² on test, cost map from tile features)", "",
          "| group | model | R² range | within-map R² range |", "|---|---|---|---|"]
    keys = [(f["group"], f["kind"]) for f in data[tags[0]]["surrogate"]["fits"]]
    for g, k in keys:
        r2 = [next(f["r2"] for f in data[t]["surrogate"]["fits"] if (f["group"], f["kind"]) == (g, k)) for t in tags]
        rw = [next(f["r2_within"] for f in data[t]["surrogate"]["fits"] if (f["group"], f["kind"]) == (g, k)) for t in tags]
        L.append(f"| {g} | {k} | {rng(r2)} | {rng(rw)} |")
    L += ["", "Sufficiency — the same search run on a substitute cost map:", "",
          "| cost map | Opt binary range | Exp mean range |", "|---|---|---|"]
    for key in data[tags[0]]["surrogate"]["sufficiency"]:
        ob = [data[t]["surrogate"]["sufficiency"][key]["opt_binary"] for t in tags]
        ex = [data[t]["surrogate"]["sufficiency"][key]["exp_mean"] for t in tags]
        L.append(f"| {key} | {rng(ob)} | {rng(ex)} |")

    L += ["", "## Random tile transplants (all 1 000 test maps, 3 draws per budget)", "",
          "| k | suboptimal | worsened | Exp > 1.5× own baseline | path changed |", "|---|---|---|---|---|"]
    for i, row in enumerate(data[tags[0]]["pilot"]["by_budget"]):
        get = lambda key: [data[t]["pilot"]["by_budget"][i][key] for t in tags]  # noqa: E731
        L.append(f"| {row['budget']} | {rng(get('frac_suboptimal'))} | {rng(get('frac_worsened'))} | "
                 f"{rng(get('frac_exp_up_1_5x'))} | {rng(get('frac_path_changed'))} |")

    sw = lambda *k: col("sweep", *k)  # noqa: E731
    L += ["", f"## Exhaustive single-tile transplants ({data[tags[0]]['sweep']['n_maps']} test maps, "
          f"{data[tags[0]]['sweep']['n_alterations']} edits per model)", "",
          "| quantity | range |", "|---|---|",
          f"| baseline-optimal maps | {rng(sw('n_baseline_optimal_maps'), '{:.0f}')} |",
          f"| … with ≥ 1 damaging edit | {rng(sw('frac_optimal_maps_with_damaging_tile'))} |",
          f"| median share of damaging edits per map | {rng(sw('median_frac_damaging_edits_per_map'))} |",
          f"| share of edits that need a response (old path no longer optimal) | {rng(sw('frac_needed_response'))} |",
          f"| share of edits that change the planner's path | {rng(sw('frac_path_changed'))} |",
          f"| worst Opt ratio | {rng(sw('worst_opt'))} |",
          f"| Spearman(ΔCost L1, Opt damage) | {rng(sw('spearman_dcost_vs_opt_damage'))} |",
          f"| share of ΔCost on the edited tile | {rng(sw('dcost_frac_under_flips_mean'))} |",
          f"| max ΔCost beyond the receptive field | {max(sw('max_dcost_outside_receptive_field')):.1e} |",
          "", "Damage by location of the edited tile (baseline-optimal maps):", "",
          "| location | share damaging | costlier edit | cheaper edit |", "|---|---|---|---|"]
    for loc in [r["location"] for r in data[tags[0]]["sweep"]["by_location"]]:
        get = lambda key: [next(r[key] for r in data[t]["sweep"]["by_location"] if r["location"] == loc) for t in tags]  # noqa: E731,B023
        L.append(f"| {loc} | {rng(get('frac_damaging'))} | {rng(get('frac_damaging_costlier'))} | {rng(get('frac_damaging_cheaper'))} |")
    L += ["", "What carries the damage (damaging edits on baseline-optimal maps):", "",
          "| quantity | range |", "|---|---|",
          f"| damaging edits | {rng(sw('decomposition', 'n_damaging'), '{:.0f}')} |",
          f"| cost-ranking (exact search on the altered costs also misses the optimum) | {rng(sw('decomposition', 'cost_ranking'))} |",
          f"| search-rule (exact search on the altered costs finds it) | {rng(sw('decomposition', 'search_rule'))} |",
          f"| unprovoked (the old path was still optimal) | {rng(sw('decomposition', 'ignoring_change_was_optimal'))} |"]

    L += ["", "## L∞ pixel perturbations (true costs unchanged)", "",
          f"{data[tags[0]]['linf']['n_instances']} test maps; PGD with {data[tags[0]]['linf']['pgd_steps']} steps "
          "on upstream's training loss.", "",
          "| ε (×1/255) | method | Opt binary | optimal maps broken | worsened | path changed |",
          "|---|---|---|---|---|---|"]
    for i, row in enumerate(data[tags[0]]["linf"]["rows"]):
        get = lambda key: [data[t]["linf"]["rows"][i][key] for t in tags]  # noqa: E731,B023
        L.append(f"| {row['eps_255']} | {row['method']} | {rng(get('opt_binary'))} | "
                 f"{rng(get('frac_optimal_maps_broken'))} | {rng(get('frac_worsened'))} | {rng(get('frac_path_changed'))} |")

    path = config.RESULTS_DIR / "warcraft.md"
    path.write_text("\n".join(L) + "\n")
    print(f"wrote {path.relative_to(config.REPO_ROOT)} and results/warcraft.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
