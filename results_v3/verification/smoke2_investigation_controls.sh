#!/usr/bin/env bash
# Controls for the v2-vs-v3 differences of the extended smoke test (ssl4polyp-ft, k = 1, seed 13).
#   repeat_*  identical repeat of SUP-imnet sun_full on GPU 1 (determinism)
#   notf32_*  sun_full + E3 + E4 s50 with TF32 off on GPU 0 (pure numerical perturbation)
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUBLAS_WORKSPACE_CONFIG=:4096:8
R=results_v3/investigation
C=checkpoints_v3/investigation/classification/k1

run() {
  local gpu=$1 name=$2
  shift 2
  local s
  s=$(date +%s)
  CUDA_VISIBLE_DEVICES=$gpu python -m ssl4polyp.classification.train_classification "$@" > "$R/$name.stdout" 2>&1
  echo "$name gpu=$gpu elapsed_s=$(( $(date +%s) - s ))" >> "$R/timings.log"
}

chain() {
  local gpu=$1 model=$2 cfg=$3 tag=$4
  shift 4
  local d=$C/${tag}_sun_full_${model}_seed13 p
  run "$gpu" "${tag}_sun_full_$model" --exp-config "$cfg" --model-key "$model" --seed 13 \
    --roots data/roots.json --output-dir "$d" "$@"
  p=$(find "$d" -maxdepth 2 -name "*_SUNFull_s13.pth" -type l | head -n1)
  run "$gpu" "${tag}_e3_$model" --exp-config exp/exp5a.yaml --model-key "$model" --seed 13 \
    --parent-checkpoint "$p" --roots data/roots.json --output-dir "$C/${tag}_e3_${model}_seed13" "$@"
  run "$gpu" "${tag}_e4_$model" --exp-config exp/exp5c/budgets/s50.yaml --model-key "$model" --seed 13 \
    --parent-checkpoint "$p" --roots data/roots.json --output-dir "$C/${tag}_e4_s50_${model}_seed13" "$@"
}

run 1 repeat_sun_full_sup_imnet --exp-config exp/exp1.yaml --model-key sup_imnet --seed 13 \
  --roots data/roots.json --output-dir "$C/repeat_sun_full_sup_imnet_seed13" &
(
  chain 0 sup_imnet exp/exp1.yaml notf32 --override tf32.matmul=false tf32.cudnn=false
  chain 0 ssl_colon exp/exp2.yaml notf32 --override tf32.matmul=false tf32.cudnn=false
) &
wait
echo "ALL_DONE" >> "$R/timings.log"
