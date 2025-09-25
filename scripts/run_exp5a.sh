#!/usr/bin/env bash

set -euo pipefail

ROOTS=${ROOTS:-data/roots.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-checkpoints/classification}
SEEDS=${SEEDS:-42 47 13}
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
    out_dir="${OUTPUT_ROOT}/exp5a_${model}_seed${seed}"
    python -m ssl4polyp.classification.train_classification \
      --exp-config exp/exp5a.yaml \
      --model-key "${model}" \
      --seed "${seed}" \
      --roots "${ROOTS}" \
      --output-dir "${out_dir}" "${@}"
  done
done
