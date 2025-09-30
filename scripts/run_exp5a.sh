#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXP_CONFIG=${EXP_CONFIG:-exp/exp5a.yaml}
ROOTS=${ROOTS:-data/roots.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-checkpoints/classification}
PARENT_ROOT=${PARENT_ROOT:-checkpoints/classification}
DEFAULT_SEEDS=$("${SCRIPT_DIR}/print_config_seeds.py" "${EXP_CONFIG}")
SEEDS=${SEEDS:-${DEFAULT_SEEDS}}
MODELS=${MODELS:-sup_imnet ssl_imnet ssl_colon}

# Canonical SUN fine-tuning checkpoints must be available prior to running this
# script. The expected layout is:
#   ${PARENT_ROOT}/sun_baselines/exp1_sup_imnet_seed{seed}/sup_imnet__SUNFull_s{seed}.pth
#   ${PARENT_ROOT}/sun_baselines/exp1_ssl_imnet_seed{seed}/ssl_imnet__SUNFull_s{seed}.pth
#   ${PARENT_ROOT}/sun_baselines/exp2_ssl_colon_seed{seed}/ssl_colon__SUNFull_s{seed}.pth
# for each seed used below. If the layout differs, update PARENT_ROOT or place
# the checkpoints under the dataset-specific subdirectory before launching.

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
    parent_dir=""
    case "${model}" in
      sup_imnet)
        parent_dir="sun_baselines"
        parent_rel="exp1_sup_imnet_seed${seed}/sup_imnet__SUNFull_s${seed}.pth"
        ;;
      ssl_imnet)
        parent_dir="sun_baselines"
        parent_rel="exp1_ssl_imnet_seed${seed}/ssl_imnet__SUNFull_s${seed}.pth"
        ;;
      ssl_colon)
        parent_dir="sun_baselines"
        parent_rel="exp2_ssl_colon_seed${seed}/ssl_colon__SUNFull_s${seed}.pth"
        ;;
      *)
        echo "Unknown model '${model}' requested; cannot resolve parent checkpoint." >&2
        exit 1
        ;;
    esac
    if [[ -n "${parent_dir}" ]]; then
      parent_ckpt="${PARENT_ROOT}/${parent_dir}/${parent_rel}"
    else
      parent_ckpt="${PARENT_ROOT}/${parent_rel}"
    fi
    if [[ ! -f "${parent_ckpt}" ]]; then
      cat >&2 <<EOF
Error: expected parent checkpoint '${parent_ckpt}' not found.
Each SUN checkpoint should follow the layout:
  \${PARENT_ROOT}/sun_baselines/exp{N}_<model>_seed{seed}/<model>__SUNFull_s{seed}.pth
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
