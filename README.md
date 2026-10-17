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

## Repository layout (planned)

```
.
├── README.md
├── docs/
│   ├── proposal.md            # answers to the course questionnaire (PT-BR)
│   └── references.md
├── neural-astar/              # git submodule: omron-sinicx/neural-astar (minimal branch)
├── data/
│   ├── mazes_032_moore_c8.npz # from the submodule / planning-datasets
│   └── generated/             # extra instances from planning-datasets scripts
├── src/
│   ├── features.py            # geometric feature library (E1)
│   ├── surrogate.py           # additive model fit + residual analysis (E1)
│   ├── attribution.py         # IG / Grad-CAM hooks (E1)
│   ├── perturb.py             # cell-flip search: gradient-guided, exhaustive, greedy, random (E2)
│   ├── swap.py                # 2×2 cost/obstacle cross runs (E3)
│   ├── detect.py              # residual-based detector + Mahalanobis baseline (E4)
│   └── metrics.py             # Opt, Exp, Hmean, ΔOpt, ΔExp, ΔCost
├── notebooks/
│   ├── 00_reproduce.ipynb
│   ├── 01_cost_map_analysis.ipynb
│   ├── 02_perturbations.ipynb
│   └── 03_mediation.ipynb
├── results/
└── requirements.txt
```

---

## Setup

Tested on Ubuntu ≥ 20.04 / WSL2, Python ≥ 3.9. Planning runs on CPU; a single free-tier GPU (Colab/Kaggle T4) is enough for training. Total GPU budget for the whole project is well under 10 hours.

```bash
git clone --recursive https://github.com/<org>/<repo>.git
cd <repo>

python -m venv .venv
source .venv/bin/activate

# Neural A* (pinned to the minimal branch)
pip install -e "./neural-astar[dev]"

# project dependencies
pip install -r requirements.txt
```

### Data

- **Mazes (32×32).** `mazes_032_moore_c8.npz` ships with the Neural A\* repo: 800 train / 100 val / 100 test instances. Each instance includes the binary map, the one-hot goal map, optimal directions, and the **true geodesic distance of every cell** — the latter is the ground truth for the E1 probes. Extra instances are generated with [`planning-datasets`](https://github.com/omron-sinicx/planning-datasets).
- **WarCraft 12×12 (conditional).** Download `warcraft_maps.tar.gz` from the [Blackbox Combinatorial Solvers page](https://edmond.mpdl.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.YJCQ5S) and extract `12x12/` to the repo root. True per-tile costs are available in the dataset. Only used if E1–E3 on mazes are finished before stage 3.

### Running

```bash
# E0 — reproduce
python neural-astar/scripts/train.py            # or use the pretrained checkpoint
python -m src.metrics --ckpt <path> --split test

# E1 — cost map analysis
python -m src.features  --out results/features.npz
python -m src.surrogate --features results/features.npz --ckpt <path>

# E2 — perturbations
python -m src.perturb --ckpt <path> --budget 1 2 4 8 --method exhaustive greedy random \
                      --objective optimality efficiency --out results/perturb/

# E3 — swap design
python -m src.swap --ckpt <path> --perturb results/perturb/ --out results/swap/
```

Scripts and flags are placeholders until stage 2; see the notebooks for the current state.

---

## Metrics

- **Opt** — fraction of instances where the found path is optimal (or path-cost ratio to the true optimum *of the altered map*).
- **Exp** — reduction in expanded nodes relative to classical A\*.
- **Hmean** — harmonic mean of Opt and Exp (as in the paper).
- **ΔOpt / ΔExp** — change under perturbation, always measured against the exact solver on the altered map.
- **ΔCost** — L1 change in the cost map and its spatial localization (under the flipped cells vs. spread).
- **Mediation fraction** — share of ΔOpt / ΔExp reproduced by each off-diagonal swap condition.

A perturbation counts as an *error* only when Neural A\* deviates from the altered map's optimum or expands substantially more than classical A\* does on that same map. Classical A\* replanning correctly is not an error.

---

## Roadmap (course stages)

- [ ] **Stage 1 — Proposal** (16 Oct 2026): question, hypothesis, design. ✔ submitted
- [ ] **Stage 2:** E0 reproduced; E1 results; perturbation harness working for at least one objective; confirm in `differentiable_astar.py` that cost map and obstacle mask are separate inputs.
- [ ] **Stage 3:** E2 + E3 complete on mazes; three training seeds; WarCraft started if time allows.
- [ ] **Final:** E4 if reached; write-up with explicit negative results.

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
