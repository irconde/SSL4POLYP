#!/usr/bin/env bash
# Campaign v3, phase 0 smoke test (spec v3.1, section 5) on the current packs (partition k = 1):
#   a) sun_full with SSL-colon, seed 13
#   b) E3 (zero-shot on polypgen_clean_test) with that model and its τ
#   c) E4 at 50 frames with the four arms (A-D)
# One GPU (GPU=1 by default). Weights go to checkpoints_v3/, run records to results_v3/smoke/.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${REPO_ROOT}"

GPU=${GPU:-1}
SEED=${SEED:-13}
MODEL=${MODEL:-ssl_colon}
PARTITION=${PARTITION:-k1}
CKPT_ROOT=${CKPT_ROOT:-checkpoints_v3/classification/${PARTITION}}
RESULTS_ROOT=${RESULTS_ROOT:-results_v3/smoke/${PARTITION}}
ROOTS=${ROOTS:-data/roots.json}

export CUDA_VISIBLE_DEVICES=${GPU}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "${RESULTS_ROOT}"

run() {
  local name=$1; shift
  local start end
  start=$(date +%s)
  echo "[$(date -Is)] start ${name}" | tee -a "${RESULTS_ROOT}/timings.log"
  python -m ssl4polyp.classification.train_classification "$@" > "${RESULTS_ROOT}/${name}.stdout" 2>&1
  end=$(date +%s)
  echo "[$(date -Is)] end ${name} elapsed_s=$((end - start))" | tee -a "${RESULTS_ROOT}/timings.log"
}

SUN_DIR="${CKPT_ROOT}/sun_full_${MODEL}_seed${SEED}"
run "a_sun_full_${MODEL}_seed${SEED}" \
  --exp-config exp/exp2.yaml --model-key "${MODEL}" --seed "${SEED}" \
  --roots "${ROOTS}" --output-dir "${SUN_DIR}"

PARENT=$(find "${SUN_DIR}" -maxdepth 2 -name "*_SUNFull_s${SEED}.pth" -type l | head -n 1)
if [[ -z "${PARENT}" ]]; then
  echo "Parent checkpoint pointer not found under ${SUN_DIR}" >&2
  exit 1
fi

run "b_e3_${MODEL}_seed${SEED}" \
  --exp-config exp/exp5a.yaml --model-key "${MODEL}" --seed "${SEED}" \
  --parent-checkpoint "${PARENT}" --roots "${ROOTS}" \
  --output-dir "${CKPT_ROOT}/e3_${MODEL}_seed${SEED}"

run "c_e4_s50_${MODEL}_seed${SEED}" \
  --exp-config exp/exp5c/budgets/s50.yaml --model-key "${MODEL}" --seed "${SEED}" \
  --parent-checkpoint "${PARENT}" --roots "${ROOTS}" \
  --output-dir "${CKPT_ROOT}/e4_s50_${MODEL}_seed${SEED}"

# Copy the run records (everything except weights and TensorBoard events) to results_v3.
rsync -a --exclude '*.pth' --exclude 'tb/' "${CKPT_ROOT}/" "${RESULTS_ROOT}/runs/"
