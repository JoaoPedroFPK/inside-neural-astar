# Convenience targets. Everything runs inside the project venv (.venv).
PY := .venv/bin/python

.PHONY: help env lock test test-env e0 step2 pilot probe features spotcheck figures surrogate sweep swap attribution budgets families damaging train seed-pipeline verify seeds warcraft-data warcraft-train warcraft detector gradient adaptive pipeline clean

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

figures:  ## README figures: dataset samples and the search animation (~6 s)
	$(PY) scripts/render_readme_figures.py --seed 0

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

train:  ## PLAN.md step 10: train three encoders (~30 min each on CPU)
	$(PY) scripts/train_encoders.py --family mazes --seeds 0 1 2 --epochs 50

seed-pipeline:  ## per-model features + sweep + swap, e.g. make seed-pipeline TAG=mazes_seed0
	bash scripts/seed_pipeline.sh $(TAG)

verify:  ## check that figures quoted in the write-ups match the stored results
	$(PY) scripts/verify_reported_numbers.py

seeds:  ## PLAN.md step 10: compare trained seeds and report dispersion
	$(PY) scripts/compare_seeds.py --tags "" mazes_seed0 mazes_seed1 mazes_seed2 \
	    --checkpoints "" results/models/mazes_seed0 results/models/mazes_seed1 results/models/mazes_seed2 \
	    --shared-flips results/manifest.parquet

warcraft-data:  ## PLAN.md step 11: fetch WarCraft maps (0.9 GB, CC BY 4.0), keep only 12x12
	mkdir -p data/warcraft
	curl -L --fail -o data/warcraft/warcraft_maps.tar.gz https://edmond.mpg.de/api/access/datafile/102059
	test "$$(md5 -q data/warcraft/warcraft_maps.tar.gz 2>/dev/null || md5sum data/warcraft/warcraft_maps.tar.gz | cut -d' ' -f1)" = acea5ea60a47664ff189923a84814e96
	tar -xzf data/warcraft/warcraft_maps.tar.gz -C data/warcraft --strip-components=1 warcraft_shortest_path_oneskin/12x12
	rm data/warcraft/warcraft_maps.tar.gz

warcraft-train:  ## PLAN.md step 11: three WarCraft encoders (~80 min each on CPU)
	$(PY) scripts/train_encoders.py --family warcraft --seeds 0 1 2 --epochs 50

warcraft:  ## PLAN.md step 11: every WarCraft stage per model, then the report (~45 min per model)
	for s in 0 1 2; do $(PY) scripts/warcraft_study.py --tag warcraft_seed$$s --stage all || exit 1; done
	$(PY) scripts/warcraft_report.py --tags warcraft_seed0 warcraft_seed1 warcraft_seed2

detector:  ## PLAN.md step 12: the altered-map detector on every condition, with the CNN control (~2 min)
	$(PY) scripts/e4_detector.py --grid-cnn

gradient:  ## PLAN.md step 13: gradient-guided flip attack vs random pool and sweep-seeded greedy (~45 min)
	$(PY) scripts/gradient_attack.py --split test --seed 0

adaptive:  ## PLAN.md step 13: detector-aware attack on step 12's surviving claim (~30 min)
	$(PY) scripts/adaptive_attack.py

pipeline:  ## every experiment in dependency order (~1h15 on a laptop CPU)
	@echo "==> step 0: reproduction"      && $(MAKE) --no-print-directory e0
	@echo "==> step 2: solver gate"       && $(MAKE) --no-print-directory step2
	@echo "==> step 3: random pilot"      && $(MAKE) --no-print-directory probe pilot
	@echo "==> step 4: feature library"   && $(MAKE) --no-print-directory features spotcheck
	@echo "==> step 5: surrogate"         && $(MAKE) --no-print-directory surrogate
	@echo "==> step 6: exhaustive sweep"  && $(MAKE) --no-print-directory sweep
	@echo "==> step 6: damaging cells"    && $(MAKE) --no-print-directory damaging
	@echo "==> step 7: mediation swap"    && $(MAKE) --no-print-directory swap
	@echo "==> step 7b: map families"     && $(MAKE) --no-print-directory families
	@echo "==> step 8: attribution"       && $(MAKE) --no-print-directory attribution
	@echo "==> step 9: greedy budgets"    && $(MAKE) --no-print-directory budgets
	@echo "==> checking the quoted figures" && $(MAKE) --no-print-directory verify

clean:  ## remove caches (keeps .venv and results/)
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache
