#!/usr/bin/env bash
# Campaign v3, phase 0 smoke test on the current packs (partition k = 1), one seed:
#   a) sun_full for each model
#   b) E3 (zero-shot on polypgen_clean_test) with each sun_full model and its τ
#   c) E4 at 50 frames with the four arms (A-D) for each model
# Runs in the ssl4polyp-ft environment on one GPU (GPU=1 by default), sequentially.
# Weights go to checkpoints_v3/<tag>/, run records to results_v3/<tag>/.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${REPO_ROOT}"

GPU=${GPU:-1}
SEED=${SEED:-13}
MODELS=${MODELS:-"sup_imnet ssl_imnet ssl_colon"}
PARTITION=${PARTITION:-k1}
SMOKE_TAG=${SMOKE_TAG:-smoke2}
CKPT_ROOT=${CKPT_ROOT:-checkpoints_v3/${SMOKE_TAG}/classification/${PARTITION}}
RESULTS_ROOT=${RESULTS_ROOT:-results_v3/${SMOKE_TAG}/${PARTITION}}
ROOTS=${ROOTS:-data/roots.json}

python -c "import timm, sys; sys.exit(0 if timm.__version__ == '0.4.12' else 'timm 0.4.12 required: activate ssl4polyp-ft')"

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

for MODEL in ${MODELS}; do
  # Same sun_full configs as the v2 parents: exp1 for the ImageNet regimes, exp2 for SSL-colon.
  case "${MODEL}" in
    ssl_colon) SUN_CONFIG=exp/exp2.yaml ;;
    *) SUN_CONFIG=exp/exp1.yaml ;;
  esac
  SUN_DIR="${CKPT_ROOT}/sun_full_${MODEL}_seed${SEED}"
  run "a_sun_full_${MODEL}_seed${SEED}" \
    --exp-config "${SUN_CONFIG}" --model-key "${MODEL}" --seed "${SEED}" \
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
done

# Copy the run records (everything except weights and TensorBoard events) to results_v3.
rsync -a --exclude '*.pth' --exclude 'tb/' "${CKPT_ROOT}/" "${RESULTS_ROOT}/runs/"
