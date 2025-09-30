#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXP_CONFIG=${EXP_CONFIG:-exp/exp4.yaml}
ROOTS=${ROOTS:-data/roots.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-checkpoints/classification}
DEFAULT_SEEDS=$("${SCRIPT_DIR}/print_config_seeds.py" "${EXP_CONFIG}")
SEEDS=${SEEDS:-${DEFAULT_SEEDS}}
PERCENTS=${PERCENTS:-5 10 25 50 100}
MODEL=${MODEL:-ssl_imnet}

python - <<'PY'
import torch

if torch.cuda.is_available():
    count = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(count)]
    devices = ", ".join(names)
    print(f"Detected {count} CUDA device(s): {devices}")
else:
    print("No CUDA devices detected; training will run on CPU.")
PY

for seed in ${SEEDS}; do
  for pct in ${PERCENTS}; do
    out_dir="${OUTPUT_ROOT}/exp4_${MODEL}_seed${seed}_p${pct}"
    python -m ssl4polyp.classification.train_classification \
      --exp-config "${EXP_CONFIG}" \
      --model-key "${MODEL}" \
      --seed "${seed}" \
      --roots "${ROOTS}" \
      --override dataset.percent="${pct}" dataset.seed="${seed}" \
      --output-dir "${out_dir}" "${@}"
  done
done
