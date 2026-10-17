# How do small changes in a map affect a learned path planner?

Interpretability and robustness analysis of **Neural A\*** (Yonetani et al., ICML 2021) — a semester project for the AI Interpretability course, PPGC/UFRGS, 2026/2.

> **Como pequenas mudanças em um mapa afetam um planejador de rotas aprendido?**

---

## TL;DR

Neural A\* is a two-stage planner: a convolutional **encoder** turns an obstacle map (plus start and goal) into a per-cell **cost map** (the *guidance map*), and a **differentiable A\*** searches on that cost map. Only the encoder is learned; the search is fixed. The cost map is therefore an explicit, human-readable bottleneck between a neural network and a classical algorithm.

We use that bottleneck to answer two questions:

1. **What does the learned cost map encode?** Is it a smoothed version of known geometric heuristics (distance to goal, proximity to obstacles), or something else — e.g. a prediction of where the path goes?
2. **When a tiny change to the map (flipping 1–8 cells) makes the planner fail, where does the failure come from?** From the encoder producing distorted costs (*encoder-mediated*), or from the search amplifying almost-correct costs (*search-mediated*)?

The second question is answered with a **swap design**: because the search takes the cost map and the obstacle mask as separate inputs, we can run it on all four combinations of {original, altered} costs × {original, altered} obstacles and measure how much of the failure each off-diagonal condition reproduces.

Classical A\* on the same altered map is the control throughout: it is exact, so its answer is by definition the *correct* response to the change.

---

## Hypothesis

The learned cost map approximates two geometric quantities — distance to goal and proximity to obstacles — in a smoothed form, so a single altered cell distorts the cost over a whole neighbourhood. Large distortions steer the search to a worse path (encoder-mediated); small distortions only break ties between near-equivalent routes and inflate the number of expanded nodes without changing the path (search-mediated). Each kind of damage — path quality vs. search efficiency — should have a dominant route.

**Falsifiers.** H1 fails if a simple additive model on geometric features explains the cost map poorly and the residual is spatially unstructured. H2 fails if neither swap condition alone reproduces the damage (interaction), or if Neural A\* turns out to be as robust as classical A\* at the budgets tested (no damage to decompose). Both outcomes are reportable.

**Main competing hypothesis.** The cost map is a *path-prediction map*, not a heuristic: the encoder is trained only so that the search history matches the reference path, so it may learn to "paint" the corridor the path goes through. A path map also correlates with distance to goal, so a good surrogate fit is not enough to distinguish the two — see *Validity risks* below.

---

## Experiments

| ID | Goal | Tests |
|----|------|-------|
| **E0** | Reproduce Neural A\* on the 32×32 maze benchmark (Opt, Exp, Hmean within tolerance). Set up attribution hooks. | — |
| **E1** | Characterize the cost map: fit an additive model on geometric features (geodesic & Euclidean distance to goal, distance to nearest obstacle, local obstacle density, corridor width); residual analysis by local topology (junction / corridor / dead end / open area); Integrated Gradients from cost cells back to the input; controlled map edits. | H1 |
| **E2** | Perturbations: flip 1–8 cells, chosen by gradient-guided search through the differentiable module (exhaustive for 1–2 cells, greedy for larger budgets), reachability preserved; random flips as baseline. Two objectives: degrade path optimality / degrade search efficiency. Compare against classical A\* on the same altered maps. | premise |
| **E3** | Mediation via the 2×2 swap design (see below). Metrics: fraction of ΔOpt and ΔExp reproduced by each off-diagonal cell; ΔCost (L1 and spatial localization). | H2 |
| **E4** *(stretch)* | Detection: do attacked inputs produce cost maps whose residual from the E1 surrogate is anomalous? Baseline: Mahalanobis distance on encoder activations. | H3 |

### The swap design (E3)

|                     | original obstacles              | altered obstacles              |
|---------------------|---------------------------------|--------------------------------|
| **original costs**  | baseline (no failure)           | *search-only* effect           |
| **altered costs**   | *encoder-only* effect           | full perturbation (failure)    |

If the failure reappears with altered costs on original obstacles, the encoder's reading of the map is what broke. If it reappears with original costs on altered obstacles, the search amplified a small cost error. If it only appears with both, the answer is "interaction".

### Validity risks we are aware of

- **The swap creates off-distribution pairs.** Costs computed from the altered map, applied to the original obstacles, is a configuration the encoder never produced in training; the search may react to the *incoherence*, not to the "encoder-only" effect. Alternative: intervene inside the encoder with linear probes and patch only the direction associated with the geometric quantity of interest.
- **Surrogate fit is correlation, not mechanism.** A path-prediction map also correlates with distance to goal. Alternative: a *sufficiency test* — replace the encoder with a fixed geometric heuristic built from the same features and check whether performance and fragility are preserved.
- **Attack selection is biased toward the encoder.** Gradient-guided flips are, by construction, cells the encoder is sensitive to. Random flips and natural perturbations (sensor noise, shifted obstacles) are the controls.
- **Mazes may not be representative.** 32×32 mazes with width-1 corridors are full of near-ties, which favours the search route. Alternative: repeat on the building-floorplan and city maps from the original `icml2021` branch, and on the WarCraft raw-image task.

---

## Status — 2026-10-17

**Steps 0–9 of the plan are done**, plus the cross-family validity check.
The environment is pinned and asserted; the published numbers are reproduced;
the structural premise of the mediation design is verified in code and under
test; the exact solver reproduces the published search exactly; every one of
the 102 198 single-cell map alterations has been evaluated; the damage has
been decomposed into its two routes at four attack budgets; the attribution
methods have been tested against a bound the architecture makes exact; and the
findings have been checked against all eight map families.

Step 10 (three training seeds) is running. Everything is reproducible with the
`make` targets below — `make help` lists them with their runtimes.

> The research plan, the verification record (every premise checked in code,
> with file and line references), the decision log and the per-step write-ups
> are kept as working notes outside this repository. Code comments and the
> generated reports refer to them by name; every claim they support is
> re-checkable here by running the test suite and the `make` targets below.

### E0 — reproduction on the 100 test mazes (CPU, ~2 s)

| | Opt | Exp | Hmean |
|---|---|---|---|
| Paper, Table 1 (MP dataset, Neural A\*) | 87.7 % | 40.1 % | 52.0 % |
| **Ours**, upstream start protocol | **84.0 %** | **43.5 %** | **57.3 %** |
| Ours, fixed starts (seed 0) | 83.0 % | 43.7 % | 57.3 % |

Mean expansions 60.7 (Neural A\*) vs 133.4 (classical A\*). Within the few-point
tolerance step 0 asks for. Caveats recorded rather than smoothed over: the
paper's row aggregates eight environment types (no maze-only breakdown is
published), and the checkpoint is the one shipped on the `minimal` branch, not
the paper's model.

### Three findings that changed the plan

1. **The swap design works as specified, with no wrapper needed.** The search
   module takes `(cost_maps, start_maps, goal_maps, obstacles_maps)` separately
   and uses the obstacle map *only* as a mask on candidate neighbours. Costs
   from one instance and obstacles from another plan a valid path
   (`tests/test_swap_possible.py`).
2. **Every move costs 1, diagonals included** — not √2, as the project notes
   originally assumed. Both upstream A\* implementations and the dataset
   generator agree, and the Chebyshev heuristic is the matching metric. The
   exact solver and `Opt` use the unit-cost model.
3. **The cost map enters `f` twice**: once accumulated in `g`, charged at the
   *parent* node of each expansion, and once added wholesale to `h`. So raising
   a cell's cost both penalises paths through it and makes it look farther from
   the goal — directly relevant to how an encoder-mediated failure can arise.

### Step 2 — the exact solver is a replica of upstream's search

`src/exact_solver.py` validated with `make step2`
(`results/step2_validation.json`):

* geodesic distances equal the dataset's stored distances on **all 1 000 maps**,
  every free cell — 0 mismatches over 451 447 cells (~1 s);
* its classical-A\* expansion count and path cost match upstream
  `VanillaAstar` on **100 %** of test instances;
* fed Neural A\*'s own cost map, it reproduces the module's expansions *and*
  path length on **100 %** of them — which is what confirms the cost
  semantics (charged at the parent, and also added to `h`);
* the project's `Opt`/`Exp` metrics reproduce the E0 line exactly
  (`p_opt` 0.830, `p_exp` 0.4373, `h_mean` 0.5728).

Two things fall out of it. Mean `Opt` ratio is **1.0154** — the pretrained
planner's paths are 1.5 % above optimal on average — and **Neural A\* expands
more nodes than classical A\* on 13 of 100 instances**, a regression the
paper's `Exp` floors to zero. So the efficiency objective of E2 has headroom
before any perturbation is applied.

### Step 3 — random flips barely dent the planner; targeted flips do

`make pilot` (1 200 alterations, 19 s) and `make probe`
(3 066 exhaustive k=1 alterations, 33 s) → `results/pilot_random.md`:

| k | Opt > 1 | **newly suboptimal** | Exp > 1.5 |
|---|---------|----------------------|-----------|
| 1 | 18 % | **1 %** | 1 % |
| 2 | 19 % | **3 %** | 1 % |
| 4 | 20 % | **4 %** | 1 % |
| 8 | 22 % | **9 %** | 1 % |

The raw `Opt > 1` column is mostly the planner's standing 17 % error rate; the
damage *caused* by a random flip is the second column. At k=1 the optimum
itself never changed (0 of 300 draws), so those failures are pure planner
deviation on a map whose optimum it already matched. When damage does occur it
is not marginal — mean `Opt` 1.08, worst 1.33.

The exhaustive k=1 probe shows this is a **weak attack, not a robust planner**:
all 3 probed instances have at least one single cell that costs optimality
(worst `Opt` 1.053 / 1.120 / 1.111), but only 0.1–1.2 % of the ~1 000 candidate
cells do. Fragility is real and extremely localized, so **step 6's exhaustive
k=1 sweep is justified and necessary** (≈18 min single-process for the test
split), greedy k ∈ {2, 4, 8} is justified, and random flips stay as the control.

Two structural findings came out of it, both correcting assumptions:

* **These are thin-wall rooms, not width-1-corridor mazes** — 10.8 % obstacles,
  one-cell-thick walls, ~10-cell-wide open areas, 68 % of free cells with all
  eight neighbours free. So *no* single flip can disconnect the goal (0 in
  9 068 tried), and the two flip directions are different edits: a **pillar**
  dropped into open space (92 % of draws, 0.9 % damage) versus a **door**
  opened in a wall (8 % of draws, 4.5 % damage). The plan's tie-heaviness
  concern survives, but via open space rather than corridors, and step 4's
  topology classes needed rethinking.
* **The encoder's response to a k=1 flip is exactly local** — radius 5 (five
  3×3 convs, *not* four), `|Δcost|` exactly zero outside the 11×11 window, with
  only 1.7 % of the Δ mass on the flipped cell itself. Any behaviour change
  farther away travelled through the search, which is a sharp mediation handle
  independent of the swap design.

### Step 4 — the feature library, and an early signal against H1

`make features` (1 000 instances, 14 features + 7 auxiliary arrays, 9 s) and
`make spotcheck` → `results/e1_features_spotcheck.{png,md}`, notebook
`notebooks/01_features_spotcheck.ipynb`.

Correlation of each feature with the cost map over free, reachable cells —
pooled, and averaged within instances (they agree, so no pooling artefact):

| feature | pooled r | within-instance r |
|---------|----------|-------------------|
| `on_any_optimal_path` | **−0.327** | **−0.339** |
| `on_reference_path` (training target) | **−0.248** | **−0.248** |
| `on_solver_path` | −0.205 | −0.205 |
| `corridor_span_min` | 0.086 | 0.062 |
| `geodesic_to_goal` | 0.033 | 0.022 |
| `chebyshev_to_goal` (= the heuristic) | 0.018 | 0.020 |
| `euclidean_to_goal` | 0.007 | 0.010 |

The path indicators are the strongest linear correlates; **every
distance-to-goal feature is essentially uncorrelated with the cost map**
(|r| ≤ 0.03). Cells on optimal paths carry *lower* cost, which is what steering
a search toward them requires. That is the direction of the plan's *competing*
hypothesis — a path-prediction map — rather than H1's smoothed geometric
heuristic. It is not the verdict: these are single-feature correlations, and
step 5's joint fits (geometric-only / path-only / both, train → test R²) are
the actual test. Recorded as it came out, per the project's working rules.

The cost map's spread is almost entirely within instances (sd 0.288 vs 0.033
between), and spans 0.000–1.000, so the sigmoid is neither saturated nor
reduced to a per-instance level shift.

One definition had to change: step 5 groups residuals by local
topology, but that partition is degenerate here (70 % "open", 0 % "dead end").
`src.features.wall_zone` — distance to the nearest wall, bucketed — is balanced
(43 / 31 / 26 %) and is what step 5 will group by; both are stored.

### Step 5 — what the cost map is made of, and what of it matters

`make surrogate` → `results/e1_surrogate.{md,json,png}`. Fitted on 360 700
cells from 800 training maps, scored on 44 635 cells from 100 unseen maps.

| feature group | ridge R² | boosting R² |
|---|---|---|
| the search's own heuristic | **0.002** | — |
| distance to goal only | 0.007 | 0.074 |
| obstacle geometry only | 0.022 | 0.153 |
| **H1 as stated** (goal + obstacles) | 0.026 | **0.240** |
| path indicators only | 0.123 | 0.123 |
| **everything** | 0.151 | **0.408** |

**The encoder does not re-encode the search's heuristic** — those features
explain 0.2 % of its output, which is what the architecture predicts, since the
search already adds the Chebyshev heuristic itself. H1 is refuted *as stated*
(linear fits reach 0.026) but partly vindicated in substance (0.240
non-linearly), and the competing path-prediction account does not replace it
either (0.123 alone). The cost map is a combination, and ~59 % of it is
explained by nothing we named.

Then the test R² cannot perform — feed the surrogate's prediction to the real
search instead of the network's output:

| cost map driving the search | exactly optimal | mean Exp ratio | Hmean |
|---|---|---|---|
| the network's own | 0.830 | 0.578 | **0.573** |
| surrogate, H1 features only | 0.620 | 0.755 | 0.413 |
| **surrogate, all features** | **0.920** | 0.627 | 0.552 |
| flat costs = classical A\* | 1.000 | 1.000 | 0.000 |

Geometry alone cannot plan. Geometry plus path indicators nearly can: a
surrogate reconstructing 41 % of the cost map plans *more optimally than the
network it imitates*, buying that with search effort and landing at 96 % of its
combined score. So the unexplained 59 % is not where planning competence lives
— it is what makes the network cut corners.

That is not just "weaker guidance". Scaling the network's own cost map toward a
constant traces the weaker-guidance curve; at scale 0.50 it reaches 0.840
optimality at 0.644 of classical effort, while the surrogate reaches **0.920 at
0.627** — strictly better on both axes. The structure interpretable features
impose is genuinely better for path quality at a comparable search cost.

### Steps 6–7 — where the damage comes from

`make sweep` evaluated **102 198 single-cell alterations** across the 100 test
maps in 21 minutes, then `make swap` decomposed the worst one per map per
objective.

**86 %** of the maps that were optimal beforehand have at least one cell whose
flip costs them optimality — but only a median **0.78 %** of candidate cells
are damaging. Fragility is near-universal and very thin, which is exactly why
random perturbation finds almost nothing. The worst single flip in the split
costs 35 % extra path length.

The mediation split, over the worst flip per map:

Both objectives have to be split by failure mode first. In some alterations
the flip *opened a new shortcut*: the optimum dropped and the planner was
scored as damaged for not taking a route that exists only under the altered
mask — which the search route varies by definition. Pooling those with genuine
planner deviation inverts the reading.

| | encoder route | search route |
|---|---|---|
| path quality, pooled (88) | +0.40 | +0.40 |
| **path quality, optimum unchanged (70)** | **+0.49** | +0.28 |
| path quality, shortcut opened (18) | +0.02 | +0.84 |
| search effort, pooled (91) | +0.19 | +0.50 |
| **search effort, optimum unchanged (55)** | **+0.23** | **+0.26** |
| search effort, shortcut opened (36) | +0.14 | +0.87 |

On genuine planner deviation, **path-quality damage is encoder-mediated**
(0.49 vs 0.28, and the raw-count decomposition agrees at 0.51 vs 0.28), while
**search-effort damage is roughly even** (0.23 vs 0.26) rather than
search-dominated as the pooled number suggests.

The locality bound held on all 200 swaps: no cost-map change fell outside the
encoder's 11×11 receptive field.

### Step 9 — the split moves with the size of the attack

`make budgets`: greedy search at k ≤ 8, with the step-3 random draws as the
control.

| budget | greedy mean Opt | random mean Opt | greedy mean Exp | random mean Exp |
|---|---|---|---|---|
| 1 | 1.110 | 1.017 | 1.152 | 0.587 |
| 8 | **1.167** | 1.020 | **1.981** | 0.599 |

Targeting is worth about eight times the path damage of random flipping, and
it *inverts* the efficiency effect: a randomly perturbed map still leaves the
planner expanding 41 % fewer cells than classical A\*, while eight targeted
flips leave it expanding twice as many — up to 7.4× in the worst case. The
learned guidance stops helping and becomes a liability.

The mediation split, restricted to the clean failure mode, **moves with the
budget and crosses over**:

| budget | encoder | search | interaction |
|---|---|---|---|
| 1 | **+0.49** | +0.28 | 24 % |
| 4 | +0.42 | +0.39 | 44 % |
| 8 | +0.38 | **+0.42** | 45 % |

A single targeted cell damages the planner mainly through the encoder; eight
cells mainly through the search, with interaction rising throughout. So "which
route mediates the damage?" has no single answer — it depends on how large the
perturbation is.

### Step 8 — an attribution method that fails its own test

Because the encoder is five 3×3 convolutions, we know *exactly* which input
cells can influence a given output — ground truth that attribution work almost
never has.

| method | share of attribution mass within radius 5 | furthest non-zero cell |
|---|---|---|
| integrated gradients | **100 %** | 3.8 |
| saliency | **100 %** | 5.0 |
| Grad-CAM | **10.9 %** | 24.6 |

Grad-CAM attributes influence up to 30 cells away, where influence is
impossible. Its channel-mean gradient weighting highlights a feature wherever
it is active, so on a network with no downsampling it draws general activity
rather than what produced one value — and the picture is persuasive, because it
highlights the route band that steps 4 and 5 identified. The figures are kept
to document the failure; no conclusion rests on them.

The randomisation sanity check passes overall (r = +0.04) but not uniformly:
wall-adjacent cells average +0.33.

### Step 7b — the findings across all eight map families

Seven more 32×32 families ship with the dataset submodule. Two results are
model-independent and change how later steps are run: the plan's cell
partition has an **empty class in every family** (a cell needs seven of eight
neighbours blocked to be a "dead end", which these generators never produce),
and **no single flip can disconnect the goal** in seven of eight families.

The planner's own numbers are *transfer*, not generalisation — it was trained
on mazes, and its Hmean falls from 0.573 to 0.272 across families, so
per-family training is required before concluding anything about them.

---

## Repository layout

✅ exists · ⬜ planned (plan step in brackets)

```
.
├── README.md
├── Makefile                   ✅ make env | test | test-env | e0 | lock
├── .python-version            ✅ 3.10.18
├── requirements.txt           ✅ direct pins, one comment per non-obvious pin
├── requirements.lock.txt      ✅ full transitive closure (174 packages)
├── neural-astar/              ✅ git submodule, `minimal` branch (never edited)
│   └── planning-datasets/     ✅ nested submodule: ships the .npz data
├── data/
│   └── mazes_032_moore_c8.npz ✅ symlink into the nested submodule
├── scripts/
│   ├── setup_env.sh           ✅ idempotent, self-verifying env build
│   ├── e0_reproduce.py        ✅ [0] E0 numbers, both start protocols
│   ├── validate_step2.py      ✅ [2] solver-vs-dataset + metrics-vs-E0 gate
│   ├── pilot_random.py        ✅ [3] random-flip pilot + budget decision
│   ├── probe_exhaustive_k1.py ✅ [3] worst-case probe (also step-6 timing)
│   ├── build_features.py      ✅ [4] feature archives for all splits
│   └── plot_feature_spotcheck.py ✅ [4] spot-check figure + correlations
├── src/
│   ├── config.py              ✅ paths, device, dataset layout, seeding
│   ├── grid.py                ✅ 8-connected geometry, BFS, flips, path validation
│   ├── data.py                ✅ split loading with fixed, stored start cells
│   ├── planner.py             ✅ checkpoint loading, cost maps, the swap primitive
│   ├── metrics.py             ✅ paper metrics + Opt/Exp ratios + mediation fraction
│   ├── exact_solver.py        ✅ [2] Dijkstra ground truth + classical-A* control
│   ├── features.py            ✅ [4] geometric + path features, two partitions
│   ├── surrogate.py           ✅ [5] feature groups, fits, residuals, sufficiency
│   ├── perturb.py             ✅ [3,6,9] flips: random, exhaustive, greedy
│   ├── swap.py                ✅ [7] the 2×2 cost/obstacle harness + mediation
│   ├── attribution.py         ✅ [8] integrated gradients, Grad-CAM, sanity checks
│   └── detect.py              ⬜ [12] residual detector + Mahalanobis baseline
├── tests/                     ✅ 154 tests across 13 files, including published-tree and manifest-schema guards
├── notebooks/                 ✅ 01_features_spotcheck.ipynb (display only)
└── results/
    ├── e0_reproduce.json      ✅ [0]
    ├── step2_validation.json  ✅ [2]
    ├── pilot_random.md        ✅ [3] the step-3 table and the budget decision
    ├── pilot_random.parquet   ✅ [3] 1 200 alterations, manifest-compatible schema
    ├── probe_k1.json          ✅ [3] exhaustive k=1 on 3 instances
    ├── features_{split}.npz   ✅ [4] 14 features + 7 aux arrays, 1 000 instances
    └── e1_features_spotcheck.{png,md} ✅ [4]
```

---

## Setup

Python 3.10 (pinned in `.python-version`), CPU-only. One command from a fresh
clone:

```bash
git clone --recursive https://github.com/<org>/inside-neural-astar.git
cd inside-neural-astar
bash scripts/setup_env.sh      # or: make env
source .venv/bin/activate
```

The script fetches the submodules, installs the interpreter via pyenv if it is
missing, builds `.venv`, installs the pinned dependencies in a single pip
invocation, and then **asserts** the result with `tests/test_env.py`. Re-run it
any time; it is idempotent. `--recreate`, `--lock` and `--no-test` are
available. Every pin carries a comment in `requirements.txt`; the two that are
easy to get wrong are `numpy==1.26.4` (2.x is ABI-incompatible with torch
1.12.1, and upstream's `numpy>=1.19.2` would resolve to it) and
`setuptools==69.5.1` (≥ 81 removed the `pkg_resources` that
pytorch-lightning 1.8.5 imports at startup).

Check an environment you did not build yourself:

```bash
make test-env     # 16 assertions: interpreter, pins, data, checkpoint, clean submodule
make test         # whole suite, 154 tests, ~35 s
```

### Data

Nothing to download or generate. `data/mazes_032_moore_c8.npz` is a symlink
into the nested `planning-datasets` submodule, which ships the file (800 train
/ 100 val / 100 test, 32×32, 8-connected). Each instance carries the binary
map, the one-hot goal map, the optimal 8-direction policy, and the **true
geodesic distance of every cell** — the ground truth for the E1 probes and for
the step-2 solver test. Seven more 32×32 map families (`forest`,
`single_bugtrap`, `shifting_gaps`, …) sit next to it, which is the planned
control for "mazes may not be representative".

The pretrained maze checkpoint also ships with the submodule, so **no training
is needed for steps 0–9**.

**WarCraft 12×12 (conditional).** Download `warcraft_maps.tar.gz` from the
[Blackbox Combinatorial Solvers page](https://edmond.mpdl.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.YJCQ5S)
and extract `12x12/` to the repo root. Only if E1–E3 on mazes finish before
stage 3.

---

## Running

Available now:

```bash
make e0                 # E0: Opt/Exp/Hmean on the test split, both start protocols
make step2              # step-2 gate: solver vs dataset (1000 maps) + metrics vs E0
make pilot              # step-3 random-flip pilot (k = 1, 2, 4, 8), ~20 s
make probe              # step-3 worst-case probe: exhaustive k=1 on 3 instances
make features           # step-4 feature archives for all 1 000 instances, ~9 s
make spotcheck          # step-4 figure + correlation diagnostic
make surrogate          # step-5 fits, residuals and the sufficiency test, ~40 s
make sweep              # step-6 exhaustive k=1 sweep, ~21 min
make swap               # step-7 mediation decomposition, ~10 s
make attribution        # step-8 attribution study, ~1 min
make families           # step-7b cross-family comparison, ~4 min
make test               # the full suite
python scripts/pilot_random.py --instances 3 --timing-only   # extrapolate first
```

Planned (the module exists in the layout above only when ✅):

```bash
python -m src.features  --out results/features_test.npz          # [4]
python -m src.surrogate --features results/features_test.npz     # [5]
python -m src.perturb   --budget 1 --method exhaustive           # [6]
python -m src.swap      --perturb results/perturb/               # [7]
```

Every script seeds numpy and torch at the top and accepts `--seed`; everything
runs on CPU (`src.config.get_device()` is the only place a device is chosen).

---

## Metrics

Two families, deliberately kept apart in `src/metrics.py`:

**The paper's metrics**, replicated line for line from upstream's
`validation_step` so that E0 is comparable to published numbers:

- **Opt** — *fraction* of instances whose path is as short as classical A\*'s.
- **Exp** — *reduction* in expanded nodes vs. classical A\*, `max((E* − E)/E*, 0)`.
- **Hmean** — harmonic mean of the two.

**This project's metrics**, which is what the perturbation analysis uses:

- **Opt (ratio)** — planner path cost ÷ true optimal cost on the **same** map
  (1.0 = optimal). After a flip, always against the *altered* map's optimum.
- **Exp (ratio)** — planner expansions ÷ classical A\* expansions on the same
  map. Above 1.0 means worse than classical, which the paper's `Exp` floors away.
- **ΔCost** — L1 change in the cost map and its spatial localization (mass
  under the flipped cells vs. spread).
- **Mediation fraction** — `(M[C1O0] − M[C0O0]) / (M[C1O1] − M[C0O0])` for the
  encoder route, the same with `C0O1` for the search route. Both reported; they
  need not sum to 1.

Path costs are counted under the **unit-cost** model (every one of the eight
moves costs 1), which is what upstream and the dataset do — see
A path map with *n* cells is a path of *n − 1* moves.

A perturbation counts as an *error* only when Neural A\* deviates from the
altered map's optimum or expands substantially more than classical A\* does on
that same map. Classical A\* replanning correctly is not an error.

---

## Roadmap (course stages)

- [x] **Stage 1 — Proposal** (16 Oct 2026): question, hypothesis, design. ✔ submitted
- [ ] **Stage 2:** E0 reproduced ✅ · swap premise confirmed in `differentiable_astar.py` ✅ · exact solver + metrics [2] ✅ · random-flip pilot [3] ✅ · feature library [4] ✅ · surrogate fit [5] ✅ · perturbation harness [6] ✅
- [ ] **Stage 3:** E2 + E3 complete on mazes [6–8] ✅ · greedy budgets [9] ✅ · three training seeds [10] 🏃 · WarCraft [11] not started — it needs a third-party dataset download, so it waits for an explicit decision rather than being fetched automatically.
- [ ] **Final:** E4 if reached [12]; write-up with explicit negative results.

**Next step:** step 10 — three encoders trained from different seeds, with the
surrogate fits and the mediation decomposition repeated on each, to turn
single-model observations into claims with a dispersion attached. After that,
per-family training finishes the validity check begun in step 7b, and the
detector of E4 becomes available.

---

## Team

Group *<group name>* — <member 1>, <member 2>, <member 3>.
Course: AI Interpretability, PPGC/UFRGS.

---

## References

- Yonetani, R. et al. (2021). *Path Planning using Neural A\* Search.* ICML. [arXiv:2009.07476](https://arxiv.org/abs/2009.07476) · [code](https://github.com/omron-sinicx/neural-astar)
- Bhatt, S. et al. (2022). *[Reproducibility Report] Path Planning using Neural A\* Search.* [arXiv:2208.04153](https://arxiv.org/abs/2208.04153)
- Vlastelica, M. et al. (2020). *Differentiation of Blackbox Combinatorial Solvers.* ICLR. [arXiv:1912.02175](https://arxiv.org/abs/1912.02175)
- Tamar, A. et al. (2016). *Value Iteration Networks.* NeurIPS. [arXiv:1602.02867](https://arxiv.org/abs/1602.02867)
- Lee, L. et al. (2018). *Gated Path Planning Networks.* ICML. [arXiv:1806.06408](https://arxiv.org/abs/1806.06408)
- Zeng, W. et al. (2019). *End-to-end Interpretable Neural Motion Planner.* CVPR. [arXiv:2101.06679](https://arxiv.org/abs/2101.06679)
- Wang, Y. et al. (2018). *A Training-based Identification Approach to VIN Adversarial Examples.* [arXiv:1810.08070](https://arxiv.org/abs/1810.08070)
- Szvoren, A. et al. (2025). *Exploring Adversarial Obstacle Attacks in Search-based Path Planning for Autonomous Mobile Robots.* [arXiv:2504.06154](https://arxiv.org/abs/2504.06154)
- Geisler, S. et al. (2022). *Generalization of Neural Combinatorial Solvers Through the Lens of Adversarial Robustness.* ICLR. [arXiv:2110.10942](https://arxiv.org/abs/2110.10942)
- Ilyas, A. et al. (2019). *Adversarial Examples Are Not Bugs, They Are Features.* NeurIPS. [arXiv:1905.02175](https://arxiv.org/abs/1905.02175)
- Sundararajan, M. et al. (2017). *Axiomatic Attribution for Deep Networks.* ICML. [arXiv:1703.01365](https://arxiv.org/abs/1703.01365)
- Selvaraju, R. et al. (2017). *Grad-CAM.* ICCV. [arXiv:1610.02391](https://arxiv.org/abs/1610.02391)
- Adebayo, J. et al. (2018). *Sanity Checks for Saliency Maps.* NeurIPS. [arXiv:1810.03292](https://arxiv.org/abs/1810.03292)
- Alain, G.; Bengio, Y. (2016). *Understanding intermediate layers using linear classifier probes.* [arXiv:1610.01644](https://arxiv.org/abs/1610.01644)
- Vig, J. et al. (2020). *Investigating Gender Bias in Language Models Using Causal Mediation Analysis.* NeurIPS. [arXiv:2004.12265](https://arxiv.org/abs/2004.12265)
- Heimersheim, S.; Nanda, N. (2024). *How to use and interpret activation patching.* [arXiv:2404.15255](https://arxiv.org/abs/2404.15255)
- Lee, K. et al. (2018). *A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks.* NeurIPS. [arXiv:1807.03888](https://arxiv.org/abs/1807.03888)
- Molnar, C. (2022). *Interpretable Machine Learning*, 2nd ed. [online](https://christophm.github.io/interpretable-ml-book/)

## License

Code: MIT (see `LICENSE`). The Neural A\* submodule keeps its own license.
