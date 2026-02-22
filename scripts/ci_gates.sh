#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

# 1) Basic syntax gate
.venv/bin/python -m py_compile scripts/*.py

# 2) Family accept gate (effect-only + configured diagnostics)
.venv/bin/python scripts/check_family_accept.py \
  --agg-json runs/mc_family_optA_effect_only_test.json \
  --K 16

# 3) Direction integrity gate for new hermetic artifacts.
# Override with DIRECTION_GLOB env var if needed.
DIR_GLOB="${DIRECTION_GLOB:-runs/*_newartifact/seed*.matched_cost_summary.json}"
shopt -s nullglob
matches=($DIR_GLOB)
if [ "${#matches[@]}" -eq 0 ]; then
  echo "no newartifact files matched: $DIR_GLOB"
  shopt -u nullglob
  exit 1
fi
shopt -u nullglob
.venv/bin/python scripts/check_direction_integrity.py \
  --seed-glob "$DIR_GLOB" \
  --hash-mode both --require-alias --strict

# 4) Artifact manifest generation + validation
MANIFEST_OUT="${ARTIFACTS_MANIFEST_OUT:-runs/_artifacts_newartifact.json}"
FIG_GLOB="${ARTIFACTS_FIG_GLOB:-runs/*_paper/*.png}"
TABLE_GLOB="${ARTIFACTS_TABLE_GLOB:-runs/*_paper/*.csv}"
REQUIRE_PAPER_ASSETS="${REQUIRE_PAPER_ASSETS:-1}"

if [ "${REQUIRE_PAPER_ASSETS}" = "1" ]; then
  shopt -s nullglob
  figs=($FIG_GLOB)
  tables=($TABLE_GLOB)
  shopt -u nullglob
  if [ "${#figs[@]}" -eq 0 ]; then
    echo "no figures matched: $FIG_GLOB"
    exit 1
  fi
  if [ "${#tables[@]}" -eq 0 ]; then
    echo "no tables matched: $TABLE_GLOB"
    exit 1
  fi
fi

.venv/bin/python scripts/make_artifacts_manifest.py \
  --seed-glob "$DIR_GLOB" \
  --out "$MANIFEST_OUT" \
  --include-figures "$FIG_GLOB" \
  --include-tables "$TABLE_GLOB"

.venv/bin/python scripts/check_artifacts_manifest.py \
  --manifest "$MANIFEST_OUT" \
  --strict

# 5) Repo-level artifacts index
MANIFEST_INDEX_OUT="${ARTIFACTS_INDEX_OUT:-runs/artifacts_index.json}"
MANIFEST_INDEX_GLOB="${ARTIFACTS_INDEX_GLOB:-runs/_artifacts*.json}"
.venv/bin/python scripts/make_artifacts_index.py \
  --manifest-glob "$MANIFEST_INDEX_GLOB" \
  --out "$MANIFEST_INDEX_OUT"
