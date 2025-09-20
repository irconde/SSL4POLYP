#!/usr/bin/env bash

set -euo pipefail

ROOTS=${ROOTS:-data/roots.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-checkpoints/classification}
SEEDS=${SEEDS:-13 29 47}
SIZES=${SIZES:-50 100 200 500}
MODELS=${MODELS:-sup_imnet ssl_imnet ssl_colon}

for seed in ${SEEDS}; do
  for size in ${SIZES}; do
    for model in ${MODELS}; do
      out_dir="${OUTPUT_ROOT}/exp5c_${model}_seed${seed}_s${size}"
      python -m ssl4polyp.classification.train_classification \
        --exp-config exp/exp5c.yaml \
        --model-key "${model}" \
        --seed "${seed}" \
        --roots "${ROOTS}" \
        --override dataset.size="${size}" dataset.seed="${seed}" \
        --output-dir "${out_dir}" "${@}"
    done
  done
done
