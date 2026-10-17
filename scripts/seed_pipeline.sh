#!/usr/bin/env bash
# Repeat the per-model measurements for one trained encoder (PLAN.md step 10).
#
#   bash scripts/seed_pipeline.sh mazes_seed0
#
# Each model needs its own feature archive (the cost map is the surrogate's
# target and changes with the weights) and its own exhaustive sweep (each model
# has its own worst flips; reusing another model's would answer a different
# question). Everything is written with the tag as a suffix, so models never
# overwrite each other.
#
# Cost: about 20 seconds for the features, ~21 minutes for the sweep, and a few
# seconds for the swap.
set -euo pipefail

TAG="${1:?usage: seed_pipeline.sh <tag>, e.g. mazes_seed0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CHECKPOINT="results/models/${TAG}"
PY=".venv/bin/python"

if [ ! -d "$CHECKPOINT" ]; then
  echo "no checkpoint directory at $CHECKPOINT" >&2
  exit 1
fi

echo "==> [$TAG] feature archives"
"$PY" scripts/build_features.py --checkpoint "$CHECKPOINT" --tag "$TAG"

echo "==> [$TAG] exhaustive k=1 sweep"
"$PY" scripts/sweep_k1.py --checkpoint "$CHECKPOINT" --tag "$TAG"

echo "==> [$TAG] mediation swap"
"$PY" scripts/swap_k1.py --checkpoint "$CHECKPOINT" --tag "$TAG"

echo "==> [$TAG] done"
