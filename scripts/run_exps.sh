#!/usr/bin/env bash
set -euo pipefail

# Run experiments Exp-1..Exp-5 using manifests and a roots mapping.
# Usage: scripts/run_exps.sh MANIFEST_DIR ROOTS_JSON OUT_BASE
# MANIFEST_DIR: Directory containing manifest YAML files exp1.yaml..exp5.yaml
# ROOTS_JSON: JSON file mapping root identifiers to filesystem paths
# OUT_BASE: Base directory for experiment outputs

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 MANIFEST_DIR ROOTS_JSON OUT_BASE" >&2
  exit 1
fi

MANIFEST_DIR="$1"
ROOTS_JSON="$2"
OUT_BASE="$3"

CSV="joblist.csv"
# Initialize joblist.csv with header if it doesn't exist
if [[ ! -f "$CSV" ]]; then
  echo "exp,manifest,output_dir" > "$CSV"
fi

for EXP in 1 2 3 4 5; do
  MANIFEST_PATH="${MANIFEST_DIR}/exp${EXP}.yaml"
  OUT_DIR="${OUT_BASE}/exp${EXP}"

  echo "${EXP},${MANIFEST_PATH},${OUT_DIR}" >> "$CSV"

  # Extract CSV paths from the manifest and ensure all referenced files exist
  mapfile -t CSV_FILES < <(
    python - "$MANIFEST_PATH" <<'PY'
import sys, yaml, pathlib
manifest = pathlib.Path(sys.argv[1])
with open(manifest) as f:
    data = yaml.safe_load(f) or {}
for entry in data.values():
    if isinstance(entry, dict) and "csv" in entry:
        p = pathlib.Path(entry["csv"])
        if not p.is_absolute():
            p = manifest.parent / p
        print(p)
PY
  )
  for CSV_FILE in "${CSV_FILES[@]}"; do
    python scripts/check_paths.py "$CSV_FILE" "$ROOTS_JSON"
  done

  python Classification/train_classification.py \
    --manifest "$MANIFEST_PATH" \
    --roots "$ROOTS_JSON" \
    --output-dir "$OUT_DIR"
done
