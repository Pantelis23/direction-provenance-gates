#!/usr/bin/env bash
set -euo pipefail

# Add AtlasLM root and src to PYTHONPATH for direct imports.
ATLASLM_ROOT="/home/pantelis/Desktop/Projects/Work/AtlasLM"
ATLASLM_SRC="/home/pantelis/Desktop/Projects/Work/AtlasLM/src"

export PYTHONPATH="${ATLASLM_ROOT}:${ATLASLM_SRC}:${PYTHONPATH:-}"

echo "PYTHONPATH set to include AtlasLM." 
