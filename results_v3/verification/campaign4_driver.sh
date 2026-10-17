#!/usr/bin/env bash
# Campaign 4: SSL-imnet-matched, phases in the order of campaign 3; stop at the first failed phase or check.
set -uo pipefail
cd "$(dirname "$0")/../.."
for phase in sun_full e2e3 e1 e4; do
  echo "[$(date -Is)] phase ${phase} start"
  python scripts/v3/run_campaign.py "${phase}" --gpus "0 1" --regimes ssl_imnet_matched || { echo "[$(date -Is)] phase ${phase} FAILED (runs)"; exit 1; }
  python scripts/v3/verify_phase.py "${phase}" --regimes ssl_imnet_matched > "results_v3/campaign/${phase}_ssl_imnet_matched_verify.log" 2>&1 \
    || { echo "[$(date -Is)] phase ${phase} FAILED (verification)"; exit 1; }
  echo "[$(date -Is)] phase ${phase} verified"
done
echo "[$(date -Is)] ALL_DONE"
