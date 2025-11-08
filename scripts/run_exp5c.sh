#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

EXP_CONFIG=${EXP_CONFIG:-exp/exp5c.yaml}
BUDGET_DIR=${BUDGET_DIR:-config/exp/exp5c/budgets}
ROOTS=${ROOTS:-data/roots.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-checkpoints/classification}
REPORTING_INPUTS_ROOT=${REPORTING_INPUTS_ROOT:-results/reporting_inputs}

DEFAULT_SEEDS=$("${SCRIPT_DIR}/print_config_seeds.py" "${EXP_CONFIG}")
DEFAULT_MODELS=$("${SCRIPT_DIR}/print_config_models.py" "${EXP_CONFIG}")
SEEDS=${SEEDS:-${DEFAULT_SEEDS}}
# Override MODELS in the environment to adjust the selection; defaults track the config.
MODELS=${MODELS:-${DEFAULT_MODELS}}

if [[ ! -d "${BUDGET_DIR}" ]]; then
  echo "Budget directory '${BUDGET_DIR}' does not exist." >&2
  exit 1
fi

mapfile -t _budget_files < <(find "${BUDGET_DIR}" -maxdepth 1 -type f -name '*.yaml' | sort)

declare -A _budget_paths=()
_all_budgets=()

for budget_file in "${_budget_files[@]}"; do
  budget_name=$(basename "${budget_file}")
  budget_stub=${budget_name%.yaml}
  budget_key=${budget_stub#s}
  _budget_paths["${budget_key}"]="${BUDGET_DIR}/${budget_stub}.yaml"
  _all_budgets+=("${budget_key}")
done

if [[ ${#_all_budgets[@]} -eq 0 ]]; then
  echo "No budget configs found in '${BUDGET_DIR}'." >&2
  exit 1
fi

if [[ ${#_all_budgets[@]} -gt 1 ]]; then
  IFS=$'\n' read -r -d '' -a _all_budgets < <(printf '%s\n' "${_all_budgets[@]}" | awk 'NF' | sort -g && printf '\0')
fi

if [[ -n "${SIZES:-}" ]]; then
  IFS=' ' read -r -a _requested_budgets <<<"${SIZES//,/ }"
else
  _requested_budgets=()
fi

budgets=()
if [[ ${#_requested_budgets[@]} -gt 0 ]]; then
  declare -A _seen_requested=()
  for raw_budget in "${_requested_budgets[@]}"; do
    [[ -z "${raw_budget}" ]] && continue
    normalized=${raw_budget#s}
    if [[ -n "${_budget_paths[${normalized}]:-}" && -z "${_seen_requested[${normalized}]:-}" ]]; then
      budgets+=("${normalized}")
      _seen_requested["${normalized}"]=1
    else
      if [[ -z "${_budget_paths[${normalized}]:-}" ]]; then
        echo "Warning: budget '${raw_budget}' not found under '${BUDGET_DIR}'; skipping." >&2
      fi
    fi
  done
else
  budgets=("${_all_budgets[@]}")
fi

if [[ ${#budgets[@]} -eq 0 ]]; then
  echo "No budgets selected to run." >&2
  exit 1
fi

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
  for budget in "${budgets[@]}"; do
    budget_config="${_budget_paths[${budget}]}"
    if [[ -z "${budget_config}" ]]; then
      echo "Warning: missing config path for budget '${budget}'; skipping." >&2
      continue
    fi
    if [[ ! "${budget_config}" = /* ]]; then
      budget_config="${REPO_ROOT}/${budget_config}"
    fi
    for model in ${MODELS}; do
      out_dir="${OUTPUT_ROOT}/exp5c_${model}_seed${seed}_s${budget}"
      python -m ssl4polyp.classification.train_classification \
        --exp-config "${budget_config}" \
        --model-key "${model}" \
        --seed "${seed}" \
        --roots "${ROOTS}" \
        --output-dir "${out_dir}" "${@}"
      python -m ssl4polyp.utils.reporting_inputs \
        --run-dir "${out_dir}" \
        --exp-config "${budget_config}" \
        --reporting-root "${REPORTING_INPUTS_ROOT}"
    done
  done
done
