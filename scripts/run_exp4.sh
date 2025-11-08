#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXP_CONFIG=${EXP_CONFIG:-exp/exp4.yaml}
ROOTS=${ROOTS:-data/roots.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-checkpoints/classification}
REPORTING_INPUTS_ROOT=${REPORTING_INPUTS_ROOT:-results/reporting_inputs}
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src:${PYTHONPATH:-}"
DEFAULT_SEEDS=$("${SCRIPT_DIR}/print_config_seeds.py" "${EXP_CONFIG}")
DEFAULT_MODELS=$("${SCRIPT_DIR}/print_config_models.py" "${EXP_CONFIG}")
DEFAULT_SUBSET_SEED=$(python - "${EXP_CONFIG}" <<'PY'
import sys

from ssl4polyp.configs.layered import load_layered_config


def _pick_seed(value):
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


cfg = load_layered_config(sys.argv[1])
protocol = (cfg.get("protocol") or {})
dataset = (cfg.get("dataset") or {})

subset_seed = protocol.get("subset_seed")
if subset_seed is None:
    subset_seed = _pick_seed(protocol.get("subset_seeds"))
if subset_seed is None:
    subset_seed = _pick_seed(dataset.get("seed"))
if subset_seed is None:
    subset_seed = _pick_seed(dataset.get("seeds"))

if subset_seed is not None:
    print(subset_seed)
PY
)
SEEDS=${SEEDS:-${DEFAULT_SEEDS}}
SUBSET_SEED=${SUBSET_SEED:-${DEFAULT_SUBSET_SEED:-13}}
PERCENTS=${PERCENTS:-5 10 25 50 100}
# Override MODELS in the environment to adjust the selection; defaults track the config.
MODELS=${MODELS:-${DEFAULT_MODELS}}

python - <<'PY'
import torch

if torch.cuda.is_available():
    count = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(count)]
    devices = ", ".join(names)
    print(f"Detected {count} CUDA device(s): {devices}")
else:
    print("No CUDA devices detected; training will run on CPU.")
PY

for seed in ${SEEDS}; do
  for pct in ${PERCENTS}; do
    for model in ${MODELS}; do
      out_dir="${OUTPUT_ROOT}/exp4_${model}_seed${seed}_p${pct}"
      python -m ssl4polyp.classification.train_classification \
        --exp-config "${EXP_CONFIG}" \
        --model-key "${model}" \
        --seed "${seed}" \
        --roots "${ROOTS}" \
        --override dataset.percent="${pct}" dataset.seed="${SUBSET_SEED}" \
        --output-dir "${out_dir}" "${@}"
      python -m ssl4polyp.utils.reporting_inputs \
        --run-dir "${out_dir}" \
        --exp-config "${EXP_CONFIG}" \
        --reporting-root "${REPORTING_INPUTS_ROOT}"
    done
  done
done
