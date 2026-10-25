#!/usr/bin/env bash
# Exploratory common-lr campaign (docs/v3/plan_exploratorio_lr_comun.md): phases in the order of campaign 3;
# stop at the first failed phase or check. Separate root and records; lr override for sun_full and E1.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=<CONDA_ROOT>/envs/ssl4polyp-ft/bin/python
OPTS=(--regimes sup_imnet ssl_imnet ssl_colon --root checkpoints_v3/campaign_lr_common/classification --lr-override 1e-3 --records results_v3/campaign_lr_common)
for phase in sun_full e2e3 e1 e4; do
  echo "[$(date -Is)] phase ${phase} start"
  "$PY" scripts/v3/run_campaign.py "${phase}" --gpus "0 1" "${OPTS[@]}" || { echo "[$(date -Is)] phase ${phase} FAILED (runs)"; exit 1; }
  "$PY" scripts/v3/verify_phase.py "${phase}" "${OPTS[@]}" > "results_v3/campaign_lr_common/${phase}_verify.log" 2>&1 \
    || { echo "[$(date -Is)] phase ${phase} FAILED (verification)"; exit 1; }
  echo "[$(date -Is)] phase ${phase} verified"
done
echo "[$(date -Is)] ALL_DONE"
