#!/usr/bin/env bash

set -euo pipefail

ROOTS=${ROOTS:-data/roots.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-checkpoints/classification}
PARENT_ROOT=${PARENT_ROOT:-checkpoints/classification}
SEEDS=${SEEDS:-42 47 13}
MODELS=${MODELS:-sup_imnet ssl_imnet ssl_colon}

# Canonical SUN fine-tuning checkpoints must be available prior to running this
# script. The expected layout is:
#   ${PARENT_ROOT}/exp1_sup_imnet_seed{seed}/sup_imnet__SUNFull_s{seed}.pth
#   ${PARENT_ROOT}/exp1_ssl_imnet_seed{seed}/ssl_imnet__SUNFull_s{seed}.pth
#   ${PARENT_ROOT}/exp2_ssl_colon_seed{seed}/ssl_colon__SUNFull_s{seed}.pth
# for each seed used below.

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
        parent_rel="exp1_sup_imnet_seed${seed}/sup_imnet__SUNFull_s${seed}.pth"
        ;;
      ssl_imnet)
        parent_rel="exp1_ssl_imnet_seed${seed}/ssl_imnet__SUNFull_s${seed}.pth"
        ;;
      ssl_colon)
        parent_rel="exp2_ssl_colon_seed${seed}/ssl_colon__SUNFull_s${seed}.pth"
        ;;
      *)
        echo "Unknown model '${model}' requested; cannot resolve parent checkpoint." >&2
        exit 1
        ;;
    esac
    parent_ckpt="${PARENT_ROOT}/${parent_rel}"
    if [[ ! -f "${parent_ckpt}" ]]; then
      echo "Warning: expected parent checkpoint '${parent_ckpt}' not found." >&2
    fi
    python -m ssl4polyp.classification.train_classification \
      --exp-config exp/exp5a.yaml \
      --model-key "${model}" \
      --seed "${seed}" \
      --parent-checkpoint "${parent_ckpt}" \
      --roots "${ROOTS}" \
      --output-dir "${out_dir}" "${@}"
  done
done
