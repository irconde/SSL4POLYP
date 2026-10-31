#!/usr/bin/env bash
# E4 negative-type sensitivity (docs/v3/sensitivity_e4_negative_type_plan.md), phase 2: the 60 adaptations on the
# s200seq packs, the campaign section 6 checks, per-run records and hashed manifest (checkpoint SHA-256 included),
# deletion of the adaptation weights, then the registered analysis code. Stops at the first failed step.
set -uo pipefail
cd "$(dirname "$0")/../.."
CONDA="$(dirname "$(dirname "$CONDA_EXE")")"  # <CONDA_ROOT>
PY="${CONDA}/envs/ssl4polyp-ft/bin/python"
PYA="${CONDA}/envs/ssl4polyp-analysis/bin/python"
V=results_v3/verification
root=checkpoints_v3/sens_e4_negtype/classification
rec=results_v3/sens_e4_negtype
OPTS=(--root "${root}" --records "${rec}")
mkdir -p "${rec}/batch_records"
echo "[$(date -Is)] adaptations start"
"$PY" scripts/v3/run_campaign.py e4negtype --gpus "0 1" "${OPTS[@]}" || { echo "[$(date -Is)] FAILED (runs)"; exit 1; }
"$PY" scripts/v3/verify_phase.py e4negtype "${OPTS[@]}" > "${rec}/e4negtype_verify.log" 2>&1 \
  || { echo "[$(date -Is)] FAILED (section 6 checks)"; exit 1; }
cp "${rec}/e4negtype/verification.json" "${V}/sens_e4_negtype_verification.json"
cp "${rec}/e4negtype/events.jsonl" "${V}/sens_e4_negtype_events.jsonl"
parts=()
for k in 1 2 3 4 5; do
  t="sens_e4_negtype_k${k}"
  "$PY" scripts/v3/batch_records.py --root "${root}/k${k}/e4" --tag "${t}" --expected 12 --out "${rec}/batch_records" \
    >> "${rec}/records.log" 2>&1 || { echo "[$(date -Is)] FAILED (records k${k})"; exit 1; }
  parts+=("${t}")
done
for kind in runs results_manifest; do
  awk 'FNR == 1 && NR != 1 { next } 1' $(printf "${rec}/batch_records/%s_${kind}.csv " "${parts[@]}") > "${V}/sens_e4_negtype_${kind}.csv"
done
echo "[$(date -Is)] checked and recorded; deleting the adaptation weights"
find "${root}" -name "*.pth" -delete
echo "[$(date -Is)] $(find "${root}" -name '*.pth' | wc -l) .pth left under the root"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. "$PYA" scripts/v3/e4_negtype.py --seq-root "${root}" --out "${V}/e4_negtype" \
  > "${rec}/analysis.log" 2>&1 || { echo "[$(date -Is)] FAILED (analysis)"; exit 1; }
cp "${rec}/analysis.log" "${V}/e4_negtype/run_log.txt"
echo "[$(date -Is)] ALL_DONE"
