#!/usr/bin/env bash
# Exploratory per-budget learning-rate experiment (docs/v3/exploratory_lr_per_budget_plan.md, revision 1): E1 batches
# of one learning rate, in order. After each batch: section 6 checks, per-run records and hashed manifest
# (checkpoint SHA-256 included), then deletion of that batch's weights only. Stops at the first failed step.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY="$(dirname "$(dirname "$CONDA_EXE")")/envs/ssl4polyp-ft/bin/python"  # <CONDA_ROOT>/envs/ssl4polyp-ft
V=results_v3/verification
for spec in "1e-4:75:sup_imnet" "3e-4:150:ssl_colon ssl_imnet_matched" "1e-3:75:ssl_imnet_matched"; do
  lr=${spec%%:*}; rest=${spec#*:}; n=${rest%%:*}; regimes=${rest#*:}
  root=checkpoints_v3/exp_lr_per_budget/lr${lr}/classification
  rec=results_v3/exp_lr_per_budget/lr${lr}
  OPTS=(--regimes ${regimes} --root "${root}" --lr-override "${lr}" --records "${rec}")
  phase_dir="${rec}/e1_${regimes// /_}"
  echo "[$(date -Is)] batch lr ${lr} (${regimes}, ${n} runs) start"
  "$PY" scripts/v3/run_campaign.py e1 --gpus "0 1" "${OPTS[@]}" || { echo "[$(date -Is)] batch lr ${lr} FAILED (runs)"; exit 1; }
  "$PY" scripts/v3/verify_phase.py e1 "${OPTS[@]}" > "${rec}/e1_verify.log" 2>&1 \
    || { echo "[$(date -Is)] batch lr ${lr} FAILED (section 6 checks)"; exit 1; }
  cp "${phase_dir}/verification.json" "${V}/exp_lr_per_budget_lr${lr}_e1_verification.json"
  cp "${phase_dir}/events.jsonl" "${V}/exp_lr_per_budget_lr${lr}_e1_events.jsonl"
  "$PY" scripts/v3/batch_records.py --root "${root}" --tag "exp_lr_per_budget_lr${lr}" --expected "${n}" \
    > "${rec}/records.log" 2>&1 || { echo "[$(date -Is)] batch lr ${lr} FAILED (records)"; exit 1; }
  echo "[$(date -Is)] batch lr ${lr} verified and recorded; deleting its weights"
  find "${root}" -name "*.pth" -delete
  echo "[$(date -Is)] batch lr ${lr} done ($(find "${root}" -name '*.pth' | wc -l) .pth left)"
done
echo "[$(date -Is)] ALL_DONE"
