#!/usr/bin/env bash
set -euo pipefail

CONFIG_ROOT=$(python - <<'PY'
from ssl4polyp.configs import config_root
print(config_root())
PY
)
REPO_ROOT=$(dirname "$CONFIG_ROOT")
DEFAULT_ROOTS_JSON="${REPO_ROOT}/data/roots.json"
DEFAULT_OUT_BASE="${REPO_ROOT}/checkpoints/classification"

# Run experiments Exp-1..Exp-5 using manifests and a roots mapping.
# Usage: scripts/run_exps.sh MANIFEST_DIR [ROOTS_JSON] [OUT_BASE]
# MANIFEST_DIR: Directory containing manifest YAML files exp1.yaml..exp5.yaml
# ROOTS_JSON: JSON file mapping root identifiers to filesystem paths
#             (default: data/roots.json)
# OUT_BASE: Base directory for experiment outputs
#           (default: checkpoints/classification)

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 MANIFEST_DIR [ROOTS_JSON] [OUT_BASE]" >&2
  exit 1
fi

MANIFEST_DIR="$1"
if [[ "$MANIFEST_DIR" != /* && ! -e "$MANIFEST_DIR" ]]; then
  MANIFEST_DIR="${CONFIG_ROOT}/${MANIFEST_DIR}"
fi
ROOTS_JSON="${2:-$DEFAULT_ROOTS_JSON}"
if [[ "$ROOTS_JSON" != /* && ! -e "$ROOTS_JSON" ]]; then
  ROOTS_JSON="${CONFIG_ROOT}/${ROOTS_JSON}"
fi
OUT_BASE="${3:-$DEFAULT_OUT_BASE}"

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

  python -m ssl4polyp.classification.train_classification \
    --manifest "$MANIFEST_PATH" \
    --roots "$ROOTS_JSON" \
    --output-dir "$OUT_DIR"
done
