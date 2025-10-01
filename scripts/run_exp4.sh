#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXP_CONFIG=${EXP_CONFIG:-exp/exp4.yaml}
ROOTS=${ROOTS:-data/roots.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-checkpoints/classification}
DEFAULT_SEEDS=$("${SCRIPT_DIR}/print_config_seeds.py" "${EXP_CONFIG}")
DEFAULT_MODELS=$("${SCRIPT_DIR}/print_config_models.py" "${EXP_CONFIG}")
SEEDS=${SEEDS:-${DEFAULT_SEEDS}}
PERCENTS=${PERCENTS:-5 10 25 50 100}
# Override MODELS in the environment to adjust the selection; defaults track the config.
MODELS=${MODELS:-${DEFAULT_MODELS}}

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
    for model in ${MODELS}; do
      out_dir="${OUTPUT_ROOT}/exp4_${model}_seed${seed}_p${pct}"
      python -m ssl4polyp.classification.train_classification \
        --exp-config "${EXP_CONFIG}" \
        --model-key "${model}" \
        --seed "${seed}" \
        --roots "${ROOTS}" \
        --override dataset.percent="${pct}" dataset.seed="${seed}" \
        --output-dir "${out_dir}" "${@}"
    done
  done
done
