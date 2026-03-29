#!/usr/bin/env bash
set -euo pipefail

# ── Environment variables ──────────────────────────────────────────
# SWANLAB_API_KEY  - required, your SwanLab API key
# DATASET_ROOT     - parent directory of datasets (default: ~/Datasets)
export SWANLAB_API_KEY="${SWANLAB_API_KEY:?Please set SWANLAB_API_KEY}"
export DATASET_ROOT="${DATASET_ROOT:-$HOME/Datasets}"

# ── Optional overrides (pass as args, e.g. loader.batch_size=32) ──
# Default dataset is top_short_merged. Switch with:
#   ./train.sh dataset.dataset_name=four_types_merged
EXTRA_ARGS=("$@")

# ── Activate venv ──────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"

# ── Launch training ────────────────────────────────────────────────
echo "Dataset root: $DATASET_ROOT"
echo "Extra args: ${EXTRA_ARGS[*]:-none}"
python train.py data=lerobot "${EXTRA_ARGS[@]}"
