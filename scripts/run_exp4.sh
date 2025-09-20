#!/usr/bin/env bash

set -euo pipefail

ROOTS=${ROOTS:-data/roots.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-checkpoints/classification}
SEEDS=${SEEDS:-13 29 47}
PERCENTS=${PERCENTS:-5 10 25 50 100}
MODEL=${MODEL:-ssl_imnet}

for seed in ${SEEDS}; do
  for pct in ${PERCENTS}; do
    out_dir="${OUTPUT_ROOT}/exp4_${MODEL}_seed${seed}_p${pct}"
    python -m ssl4polyp.classification.train_classification \
      --exp-config exp/exp4.yaml \
      --model-key "${MODEL}" \
      --seed "${seed}" \
      --roots "${ROOTS}" \
      --override dataset.percent="${pct}" dataset.seed="${seed}" \
      --output-dir "${out_dir}" "${@}"
  done
done
