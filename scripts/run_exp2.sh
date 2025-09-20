#!/usr/bin/env bash

set -euo pipefail

ROOTS=${ROOTS:-data/roots.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-checkpoints/classification}
SEEDS=${SEEDS:-42 47 13}
MODELS=${MODELS:-ssl_imnet ssl_colon}

for seed in ${SEEDS}; do
  for model in ${MODELS}; do
    out_dir="${OUTPUT_ROOT}/exp2_${model}_seed${seed}"
    python -m ssl4polyp.classification.train_classification \
      --exp-config exp/exp2.yaml \
      --model-key "${model}" \
      --seed "${seed}" \
      --roots "${ROOTS}" \
      --output-dir "${out_dir}" "${@}"
  done
done
