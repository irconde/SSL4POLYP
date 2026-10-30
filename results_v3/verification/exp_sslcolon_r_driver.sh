#!/usr/bin/env bash
# Exploratory SSL-colon-r campaign (docs/v3/exploratory_sslcolon_repretrain_plan.md, revision 1): phases sun_full, e2e3,
# e1, e4 of the ssl_colon_r regime, in order. After each phase: section 6 checks, per-run records and hashed manifest
# (checkpoint SHA-256 included). E1 weights are deleted after its records; sun_full (parents of E2/E3/E4), E2, E3 and
# E4 weights are kept. Stops at the first failed step. START=<phase> resumes from that phase (finished runs are skipped
# by the launcher).
set -uo pipefail
cd "$(dirname "$0")/../.."
PY="$(dirname "$(dirname "$CONDA_EXE")")/envs/ssl4polyp-ft/bin/python"  # <CONDA_ROOT>/envs/ssl4polyp-ft
V=results_v3/verification
root=checkpoints_v3/exp_sslcolon_r/classification
rec=results_v3/exp_sslcolon_r
OPTS=(--regimes ssl_colon_r --root "${root}" --records "${rec}")
mkdir -p "${rec}/batch_records"
started=""
for spec in "sun_full:sun_full:3" "e2e3:e2 e3:3" "e1:e1:15" "e4:e4:12"; do
  phase=${spec%%:*}; rest=${spec#*:}; dirs=${rest%%:*}; per_k=${rest#*:}
  [ -n "${started}" ] || [ "${phase}" = "${START:-sun_full}" ] || continue
  started=1
  phase_dir="${rec}/${phase}_ssl_colon_r"
  echo "[$(date -Is)] phase ${phase} start"
  "$PY" scripts/v3/run_campaign.py "${phase}" --gpus "0 1" "${OPTS[@]}" || { echo "[$(date -Is)] phase ${phase} FAILED (runs)"; exit 1; }
  "$PY" scripts/v3/verify_phase.py "${phase}" "${OPTS[@]}" > "${rec}/${phase}_verify.log" 2>&1 \
    || { echo "[$(date -Is)] phase ${phase} FAILED (section 6 checks)"; exit 1; }
  cp "${phase_dir}/verification.json" "${V}/exp_sslcolon_r_${phase}_verification.json"
  cp "${phase_dir}/events.jsonl" "${V}/exp_sslcolon_r_${phase}_events.jsonl"
  # records per experiment directory and partition, merged into one runs table and one manifest per phase
  parts=()
  for d in ${dirs}; do
    for k in 1 2 3 4 5; do
      t="exp_sslcolon_r_${d}_k${k}"
      "$PY" scripts/v3/batch_records.py --root "${root}/k${k}/${d}" --tag "${t}" --expected "${per_k}" \
        --out "${rec}/batch_records" >> "${rec}/${phase}_records.log" 2>&1 \
        || [ "${phase}" = e2e3 ] || { echo "[$(date -Is)] phase ${phase} FAILED (records ${d} k${k})"; exit 1; }
      parts+=("${t}")
    done
  done
  if [ "${phase}" = e2e3 ]; then
    # E2/E3 only evaluate their sun_full parent, which lies outside the run directory: batch_records cannot hash it
    # (it still writes the runs table and manifest). The parent's SHA-256 is recomputed against the metrics JSON here.
    "$PY" - "${root}" "${V}/exp_sslcolon_r_e2e3_parent_check.csv" >> "${rec}/${phase}_records.log" 2>&1 <<'EOF' \
      || { echo "[$(date -Is)] phase e2e3 FAILED (parent checkpoint hashes)"; exit 1; }
import csv, hashlib, json, sys
from pathlib import Path
root, out = Path(sys.argv[1]), Path(sys.argv[2])
rows = []
for m in sorted(root.glob("k?/e[23]/*/*/*.metrics.json")):
    ev = json.loads(m.read_text())["evaluated_checkpoint"]
    p = (m.parent / ev["path"]).resolve()
    h = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else ""
    rows.append({"metrics_json": str(m), "parent_checkpoint": str(p.relative_to(Path.cwd())), "epoch": ev.get("epoch"),
                 "sha256": h, "matches_metrics_json": h == ev["sha256"]})
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader(); w.writerows(rows)
print(len(rows), "E2/E3 runs,", sum(r["matches_metrics_json"] for r in rows), "parent hashes match")
sys.exit(0 if len(rows) == 30 and all(r["matches_metrics_json"] for r in rows) else 1)
EOF
  fi
  for kind in runs results_manifest; do
    awk 'FNR == 1 && NR != 1 { next } 1' $(printf "${rec}/batch_records/%s_${kind}.csv " "${parts[@]}") \
      > "${V}/exp_sslcolon_r_${phase}_${kind}.csv"
  done
  if [ "${phase}" = e1 ]; then
    echo "[$(date -Is)] phase e1 verified and recorded; deleting its weights"
    find "${root}"/k?/e1 -name "*.pth" -delete
  fi
  echo "[$(date -Is)] phase ${phase} done ($(find "${root}" -name '*.pth' | wc -l) .pth under the root)"
done
echo "[$(date -Is)] ALL_DONE"
