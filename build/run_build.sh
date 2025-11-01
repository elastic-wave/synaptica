#!/usr/bin/env bash
set -euo pipefail

RECIPE="${1:-}"
if [[ "$RECIPE" == "--recipe" ]]; then
  RECIPE="${2:-}"
fi
if [[ -z "${RECIPE}" ]]; then
  echo "Usage: $0 --recipe configs/recipes/tinyllama.json"
  exit 1
fi

echo "[run_build] Using recipe: ${RECIPE}"
python build/build_engine.py --recipe "${RECIPE}"
echo "[run_build] Done."
