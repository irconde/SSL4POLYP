#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXP_CONFIG=${EXP_CONFIG:-exp/exp1_smoke.yaml}
ROOTS=${ROOTS:-data/roots.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-checkpoints/classification}
MODELS=${MODELS:-sup_imnet}
DATASET_PERCENT=${DATASET_PERCENT:-10}
DATASET_SEED=${DATASET_SEED:-13}
LIMIT_TRAIN=${LIMIT_TRAIN:-8}
LIMIT_VAL=${LIMIT_VAL:-4}
LIMIT_TEST=${LIMIT_TEST:-4}

"${SCRIPT_DIR}/print_config_seeds.py" "${EXP_CONFIG}" >/dev/null 2>&1 || true

for model in ${MODELS}; do
  out_dir="${OUTPUT_ROOT}/exp1_smoke_${model}_seed${DATASET_SEED}"
  python -m ssl4polyp.classification.train_classification \
    --exp-config "${EXP_CONFIG}" \
    --model-key "${model}" \
    --dataset-percent "${DATASET_PERCENT}" \
    --dataset-seed "${DATASET_SEED}" \
    --limit-train-batches "${LIMIT_TRAIN}" \
    --limit-val-batches "${LIMIT_VAL}" \
    --limit-test-batches "${LIMIT_TEST}" \
    --roots "${ROOTS}" \
    --output-dir "${out_dir}" "$@"
done