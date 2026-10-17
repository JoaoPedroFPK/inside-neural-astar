# Convenience targets. Everything runs inside the project venv (.venv).
PY := .venv/bin/python

.PHONY: help env lock test test-env e0 step2 pilot probe features spotcheck surrogate sweep swap attribution budgets families damaging seeds clean

help:  ## show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-10s %s\n", $$1, $$2}'

env:  ## build .venv reproducibly (submodules, interpreter, pinned deps, checks)
	bash scripts/setup_env.sh

lock:  ## rebuild the environment and refresh requirements.lock.txt
	bash scripts/setup_env.sh --lock

test-env:  ## assert the environment matches the pinned one
	$(PY) -m pytest tests/test_env.py

test:  ## run the whole test suite
	$(PY) -m pytest

e0:  ## PLAN.md step 0: reproduce the upstream maze numbers on the test split
	$(PY) scripts/e0_reproduce.py --split test --seed 0

step2:  ## PLAN.md step 2 gate: solver vs dataset (1000 maps) + metrics vs E0
	$(PY) scripts/validate_step2.py --split test --seed 0

pilot:  ## PLAN.md step 3: random-flip pilot (k=1,2,4,8 on the test split)
	$(PY) scripts/pilot_random.py --split test --seed 0

probe:  ## step-3 worst-case probe: exhaustive k=1 on 3 instances (step-6 timing)
	$(PY) scripts/probe_exhaustive_k1.py --instances 3

features:  ## PLAN.md step 4: build results/features_{split}.npz for all splits
	$(PY) scripts/build_features.py

spotcheck:  ## step-4 spot-check figure + correlation diagnostic
	$(PY) scripts/plot_feature_spotcheck.py --split test --instances 0 1 2

surrogate:  ## PLAN.md step 5: surrogate fits, residuals and the sufficiency test
	$(PY) scripts/fit_surrogate.py

sweep:  ## PLAN.md step 6: exhaustive k=1 flips on every test map (~19 min)
	$(PY) scripts/sweep_k1.py --split test --seed 0

swap:  ## PLAN.md step 7: 2x2 mediation swap on the worst k=1 flips
	$(PY) scripts/swap_k1.py --split test --seed 0

attribution:  ## PLAN.md step 8: integrated gradients + randomisation sanity check
	$(PY) scripts/attribution_study.py --per-group 5

budgets:  ## PLAN.md step 9: greedy search at k<=8 and mediation vs budget
	$(PY) scripts/greedy_budgets.py --split test --seed 0

families:  ## PLAN.md step 7b(a): maze-trained planner across all 8 map families
	$(PY) scripts/compare_families.py

damaging:  ## where the damaging cells are: join the sweep to the feature library
	$(PY) scripts/analyse_damaging_cells.py

seeds:  ## PLAN.md step 10: compare trained seeds and report dispersion
	$(PY) scripts/compare_seeds.py --tags "" mazes_seed0 mazes_seed1 mazes_seed2

clean:  ## remove caches (keeps .venv and results/)
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache
