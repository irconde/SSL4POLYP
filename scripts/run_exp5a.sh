#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXP_CONFIG=${EXP_CONFIG:-exp/exp5a.yaml}
ROOTS=${ROOTS:-data/roots.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-checkpoints/classification}
PARENT_ROOT=${PARENT_ROOT:-checkpoints/classification}
DEFAULT_SEEDS=$("${SCRIPT_DIR}/print_config_seeds.py" "${EXP_CONFIG}")
DEFAULT_MODELS=$("${SCRIPT_DIR}/print_config_models.py" "${EXP_CONFIG}")
SEEDS=${SEEDS:-${DEFAULT_SEEDS}}
# Override MODELS in the environment to adjust the selection; defaults track the config.
MODELS=${MODELS:-${DEFAULT_MODELS}}

# Canonical SUN fine-tuning checkpoints must exist prior to running this script.
# With the default experiment launchers, the expected layout is:
#   ${PARENT_ROOT}/exp1_sup_imnet_seed{seed}/sun_baselines/SUPImNet_SUNFull_s{seed}.pth
#   ${PARENT_ROOT}/exp1_ssl_imnet_seed{seed}/sun_baselines/SSLImNet_SUNFull_s{seed}.pth
#   ${PARENT_ROOT}/exp2_ssl_colon_seed{seed}/sun_baselines/SSLColon_SUNFull_s{seed}.pth
# for every seed you plan to reuse. Adjust PARENT_ROOT or the mappings below if
# your checkpoints live elsewhere.

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
  for model in ${MODELS}; do
    out_dir="${OUTPUT_ROOT}/exp5a_${model}_seed${seed}"
    case "${model}" in
      sup_imnet)
        experiment_dir="exp1_${model}_seed${seed}"
        ;;
      ssl_imnet)
        experiment_dir="exp1_${model}_seed${seed}"
        ;;
      ssl_colon)
        experiment_dir="exp2_${model}_seed${seed}"
        ;;
      *)
        echo "Unknown model '${model}' requested; cannot resolve parent checkpoint." >&2
        exit 1
        ;;
    esac
    parent_rel=$(MODEL="${model}" SEED="${seed}" EXPERIMENT_DIR="${experiment_dir}" python - <<'PY'
import os
from pathlib import Path

from ssl4polyp.classification.train_classification import (
    _canonicalize_tag,
    _compose_stem,
)

model = os.environ["MODEL"]
seed = int(os.environ["SEED"])
experiment_dir = os.environ["EXPERIMENT_DIR"]
model_tag = _canonicalize_tag(model)
stem = _compose_stem(model_tag, "SUNFull", (), seed)
print(Path(experiment_dir) / "sun_baselines" / f"{stem}.pth")
PY
)
    parent_ckpt="${PARENT_ROOT}/${parent_rel}"
    if [[ ! -f "${parent_ckpt}" ]]; then
      cat >&2 <<EOF
Error: expected parent checkpoint '${parent_ckpt}' not found.
Each SUN checkpoint should follow the layout:
  \${PARENT_ROOT}/exp{N}_<model>_seed{seed}/sun_baselines/<ModelTag>_SUNFull_s{seed}.pth
Ensure the dataset-specific directory exists and contains the required files before rerunning.
EOF
      exit 1
    fi
    python -m ssl4polyp.classification.train_classification \
      --exp-config "${EXP_CONFIG}" \
      --model-key "${model}" \
      --seed "${seed}" \
      --parent-checkpoint "${parent_ckpt}" \
      --roots "${ROOTS}" \
      --output-dir "${out_dir}" "${@}"
  done
done
