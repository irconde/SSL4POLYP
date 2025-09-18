#!/usr/bin/env bash
set -euo pipefail

CONFIG_ROOT=$(python - <<'PY'
from ssl4polyp.configs import config_root
print(config_root())
PY
)
REPO_ROOT=$(dirname "$CONFIG_ROOT")
DEFAULT_ROOTS_JSON="${REPO_ROOT}/data/roots.json"
DEFAULT_OUT_BASE="${REPO_ROOT}/checkpoints/classification"
DEFAULT_RESULTS_BASE="${REPO_ROOT}/results/classification"

# Usage: scripts/run_exps.sh EXP_CONFIG_DIR [ROOTS_JSON] [OUT_BASE] [RESULTS_BASE]
if [[ $# -lt 1 || $# -gt 4 ]]; then
  echo "Usage: $0 EXP_CONFIG_DIR [ROOTS_JSON] [OUT_BASE] [RESULTS_BASE]" >&2
  exit 1
fi

EXP_DIR="$1"
if [[ "$EXP_DIR" != /* && ! -e "$EXP_DIR" ]]; then
  EXP_DIR="${CONFIG_ROOT}/${EXP_DIR}"
fi
ROOTS_JSON="${2:-$DEFAULT_ROOTS_JSON}"
if [[ "$ROOTS_JSON" != /* && ! -e "$ROOTS_JSON" ]]; then
  ROOTS_JSON="${REPO_ROOT}/${ROOTS_JSON}"
fi
OUT_BASE="${3:-$DEFAULT_OUT_BASE}"
RESULTS_BASE="${4:-$DEFAULT_RESULTS_BASE}"

if [[ ! -d "$EXP_DIR" ]]; then
  echo "Experiment configuration directory not found: $EXP_DIR" >&2
  exit 1
fi
if [[ ! -f "$ROOTS_JSON" ]]; then
  echo "Roots mapping not found: $ROOTS_JSON" >&2
  exit 1
fi

mapfile -t EXP_CONFIGS < <(python - "$EXP_DIR" <<'PY'
import pathlib
import sys

exp_dir = pathlib.Path(sys.argv[1])
for path in sorted(exp_dir.glob("*.yaml")):
    print(path)
PY
)

if [[ ${#EXP_CONFIGS[@]} -eq 0 ]]; then
  echo "No experiment configurations found in $EXP_DIR" >&2
  exit 1
fi

for CONFIG_PATH in "${EXP_CONFIGS[@]}"; do
  EXP_NAME=$(basename "${CONFIG_PATH%.*}")
  echo "Launching experiment ${EXP_NAME}"
  mapfile -t SEEDS < <(python - "$CONFIG_PATH" <<'PY'
import sys
from ssl4polyp.configs.layered import load_layered_config

cfg = load_layered_config(sys.argv[1])
raw_seeds = cfg.get("seeds")
if raw_seeds is None:
    seed = cfg.get("seed")
    raw_seeds = [seed] if seed is not None else []
if isinstance(raw_seeds, int):
    raw_seeds = [raw_seeds]
if isinstance(raw_seeds, str):
    parts = [p for p in raw_seeds.replace(',', ' ').split() if p]
    raw_seeds = [int(p) for p in parts]
else:
    raw_seeds = [int(s) for s in raw_seeds]
for seed in raw_seeds:
    print(seed)
PY
  )
  if [[ ${#SEEDS[@]} -eq 0 ]]; then
    echo "No seeds specified for $CONFIG_PATH; skipping" >&2
    continue
  fi
  for SEED in "${SEEDS[@]}"; do
    OUT_DIR="${OUT_BASE}/${EXP_NAME}/seed${SEED}"
    mkdir -p "$OUT_DIR"
    echo "  -> seed ${SEED}"
    python -m ssl4polyp.classification.train_classification \
      --exp-config "$CONFIG_PATH" \
      --seed "$SEED" \
      --output-dir "$OUT_DIR" \
      --roots "$ROOTS_JSON"
    METRICS_SRC="${OUT_DIR}/metrics.json"
    if [[ ! -f "$METRICS_SRC" ]]; then
      echo "Metrics export missing for ${CONFIG_PATH} seed ${SEED}" >&2
      exit 1
    fi
    RESULTS_DIR="${RESULTS_BASE}/${EXP_NAME}"
    mkdir -p "$RESULTS_DIR"
    cp "$METRICS_SRC" "${RESULTS_DIR}/seed${SEED}.json"
  done
done
