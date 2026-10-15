#!/usr/bin/env bash
# SAP v3.1, section 10: validation-only learning-rate sweep on sun_full, partition k = 1.
# 3 regimes x 3 learning rates x 3 seeds = 27 runs, spread over GPUS by a shared job queue.
# The test split is never evaluated (--skip-test-eval). Run in the ssl4polyp-ft environment.
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${REPO_ROOT}"
python -c "import timm, sys; sys.exit(0 if timm.__version__ == '0.4.12' else 'timm 0.4.12 required: activate ssl4polyp-ft')"

GPUS=${GPUS:-"0 1"}
CKPT_ROOT=${CKPT_ROOT:-checkpoints_v3/lr_sweep/classification/k1}
RESULTS_ROOT=${RESULTS_ROOT:-results_v3/lr_sweep/k1}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "${RESULTS_ROOT}"
QUEUE="${RESULTS_ROOT}/queue.txt"
LOCK="${RESULTS_ROOT}/queue.lock"

: > "${QUEUE}"
for seed in 13 29 47; do
  for lr in 1.0e-3 3.0e-4 1.0e-4; do
    for model in ssl_colon ssl_imnet sup_imnet; do
      echo "${model} ${lr} ${seed}" >> "${QUEUE}"
    done
  done
done

worker() {
  local gpu=$1 line model lr seed cfg name out start
  while true; do
    line=$(flock "${LOCK}" bash -c 'l=$(head -n1 "$0"); [ -n "$l" ] && sed -i 1d "$0"; echo "$l"' "${QUEUE}")
    [ -z "${line}" ] && break
    read -r model lr seed <<< "${line}"
    case "${model}" in ssl_colon) cfg=exp/exp2.yaml ;; *) cfg=exp/exp1.yaml ;; esac
    name="lr${lr}_${model}_seed${seed}"
    out="${CKPT_ROOT}/lr${lr}/sun_full_${model}_seed${seed}"
    start=$(date +%s)
    CUDA_VISIBLE_DEVICES=${gpu} python -m ssl4polyp.classification.train_classification \
      --exp-config "${cfg}" --model-key "${model}" --seed "${seed}" --roots data/roots.json \
      --output-dir "${out}" --override "lr=${lr}" --skip-test-eval > "${RESULTS_ROOT}/${name}.stdout" 2>&1 \
      && status=ok || status=failed
    echo "${name} gpu=${gpu} ${status} start=$(date -d @${start} -Is) elapsed_s=$(( $(date +%s) - start ))" >> "${RESULTS_ROOT}/timings.log"
  done
}

for gpu in ${GPUS}; do worker "${gpu}" & done
wait
echo "ALL_DONE $(date -Is)" >> "${RESULTS_ROOT}/timings.log"
