#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXP_CONFIG=${EXP_CONFIG:-exp/exp5b.yaml}
ROOTS=${ROOTS:-data/roots.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-checkpoints/classification}
DEFAULT_SEEDS=$("${SCRIPT_DIR}/print_config_seeds.py" "${EXP_CONFIG}")
SEEDS=${SEEDS:-${DEFAULT_SEEDS}}
MODELS=${MODELS:-sup_imnet ssl_imnet ssl_colon}

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
  for model in ${MODELS}; do
    out_dir="${OUTPUT_ROOT}/exp5b_${model}_seed${seed}"
    python -m ssl4polyp.classification.train_classification \
      --exp-config "${EXP_CONFIG}" \
      --model-key "${model}" \
      --seed "${seed}" \
      --roots "${ROOTS}" \
      --output-dir "${out_dir}" "${@}"
  done
done
