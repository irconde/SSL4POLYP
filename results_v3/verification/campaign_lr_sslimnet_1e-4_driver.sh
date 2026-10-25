#!/usr/bin/env bash
# Exploratory SSL-imnet lr 1e-4 campaign (docs/v3/plan_exploratorio_lr_sslimnet.md): sun_full then E1;
# stop at the first failed phase or check. Separate root and records.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=<CONDA_ROOT>/envs/ssl4polyp-ft/bin/python
OPTS=(--regimes ssl_imnet --root checkpoints_v3/campaign_lr_sslimnet_1e-4/classification --lr-override 1e-4 --records results_v3/campaign_lr_sslimnet_1e-4)
for phase in sun_full e1; do
  echo "[$(date -Is)] phase ${phase} start"
  "$PY" scripts/v3/run_campaign.py "${phase}" --gpus "0 1" "${OPTS[@]}" || { echo "[$(date -Is)] phase ${phase} FAILED (runs)"; exit 1; }
  "$PY" scripts/v3/verify_phase.py "${phase}" "${OPTS[@]}" > "results_v3/campaign_lr_sslimnet_1e-4/${phase}_verify.log" 2>&1 \
    || { echo "[$(date -Is)] phase ${phase} FAILED (verification)"; exit 1; }
  echo "[$(date -Is)] phase ${phase} verified"
done
echo "[$(date -Is)] ALL_DONE"
