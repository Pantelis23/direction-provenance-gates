#!/usr/bin/env bash
set -euo pipefail

# Add AtlasLM root and src to PYTHONPATH for direct imports.
# Override via ATLASLM_ROOT / ATLASLM_SRC if your layout differs.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ATLASLM_ROOT="${ATLASLM_ROOT:-${REPO_ROOT}/../AtlasLM}"
ATLASLM_SRC="${ATLASLM_SRC:-${ATLASLM_ROOT}/src}"

export PYTHONPATH="${ATLASLM_ROOT}:${ATLASLM_SRC}:${PYTHONPATH:-}"

if [ ! -d "$ATLASLM_ROOT" ] || [ ! -d "$ATLASLM_SRC" ]; then
  echo "Warning: AtlasLM paths not found. Set ATLASLM_ROOT/ATLASLM_SRC as needed." >&2
fi
echo "PYTHONPATH set to include AtlasLM."
