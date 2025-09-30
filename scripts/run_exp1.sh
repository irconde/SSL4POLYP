#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXP_CONFIG=${EXP_CONFIG:-exp/exp1.yaml}
ROOTS=${ROOTS:-data/roots.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-checkpoints/classification}
DEFAULT_SEEDS=$("${SCRIPT_DIR}/print_config_seeds.py" "${EXP_CONFIG}")
SEEDS=${SEEDS:-${DEFAULT_SEEDS}}
MODELS=${MODELS:-sup_imnet ssl_imnet}

python - <<'PY'
try:
    import torch
except ModuleNotFoundError:
    print("PyTorch not installed; skipping CUDA availability check.")
else:
    try:
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            try:
                names = [torch.cuda.get_device_name(i) for i in range(count)]
                devices = ", ".join(names)
            except RuntimeError:
                devices = "unknown CUDA device(s)"
            print(f"Detected {count} CUDA device(s): {devices}")
        else:
            print("No CUDA devices detected; training will run on CPU.")
    except RuntimeError as err:
        print(f"CUDA initialization failed ({err}); training will run on CPU.")
PY

for seed in ${SEEDS}; do
  for model in ${MODELS}; do
    out_dir="${OUTPUT_ROOT}/exp1_${model}_seed${seed}"
    python -m ssl4polyp.classification.train_classification \
      --exp-config "${EXP_CONFIG}" \
      --model-key "${model}" \
      --seed "${seed}" \
      --roots "${ROOTS}" \
      --output-dir "${out_dir}" "${@}"
  done
done
