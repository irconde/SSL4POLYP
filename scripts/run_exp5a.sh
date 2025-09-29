#!/usr/bin/env bash

set -euo pipefail

ROOTS=${ROOTS:-data/roots.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-checkpoints/classification}
SEEDS=${SEEDS:-42 47 13}
MODELS=${MODELS:-sup_imnet ssl_imnet ssl_colon}
EXP_CONFIG=${EXP_CONFIG:-exp/exp5a.yaml}

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

readarray -t TARGET_META < <(EXP_CONFIG_REF="${EXP_CONFIG}" python - <<'PY'
import os
from ssl4polyp.configs.layered import load_layered_config

config = load_layered_config(os.environ["EXP_CONFIG_REF"])
targets = config.get("transfer_targets") or {}
polyp = targets.get("polypgen_clean") or {}
name = polyp.get("name") or "polypgen_clean_test"
pack = polyp.get("pack") or name
split = polyp.get("test_split") or "test"
stem = polyp.get("stem") or "polypgen"
print(name)
print(pack)
print(split)
print(stem)
PY
)

POLYP_NAME=${TARGET_META[0]}
POLYP_PACK=${TARGET_META[1]}
POLYP_SPLIT=${TARGET_META[2]}
POLYP_STEM=${TARGET_META[3]}

resolve_checkpoint() {
  local run_dir="$1"
  python - <<'PY' "$run_dir"
from pathlib import Path
import sys

run_dir = Path(sys.argv[1]).expanduser()
if not run_dir.exists():
    raise SystemExit(f"Training directory not found: {run_dir}")

candidates = sorted(run_dir.rglob("*.pth"))
if not candidates:
    raise SystemExit(f"No checkpoints found under {run_dir}")

pointer = None
for path in candidates:
    try:
        relative_parts = path.relative_to(run_dir).parts[:-1]
    except ValueError:
        relative_parts = ()
    if any(part.startswith("eval_") for part in relative_parts):
        continue
    stem = path.stem
    if stem.endswith("_last"):
        continue
    if "_e" not in stem:
        pointer = path
        break
if pointer is None:
    for path in reversed(candidates):
        try:
            relative_parts = path.relative_to(run_dir).parts[:-1]
        except ValueError:
            relative_parts = ()
        if any(part.startswith("eval_") for part in relative_parts):
            continue
        pointer = path
        break
    if pointer is None:
        pointer = candidates[-1]
print(pointer.resolve())
PY
}

for seed in ${SEEDS}; do
  for model in ${MODELS}; do
    base_dir="${OUTPUT_ROOT}/exp5a_${model}_seed${seed}"
    python -m ssl4polyp.classification.train_classification \
      --exp-config "${EXP_CONFIG}" \
      --model-key "${model}" \
      --seed "${seed}" \
      --roots "${ROOTS}" \
      --output-dir "${base_dir}" "$@"

    checkpoint_path=$(resolve_checkpoint "${base_dir}")

    sun_eval_dir="${base_dir}/eval_sun"
    python -m ssl4polyp.classification.train_classification \
      --exp-config "${EXP_CONFIG}" \
      --model-key "${model}" \
      --seed "${seed}" \
      --roots "${ROOTS}" \
      --output-dir "${sun_eval_dir}" \
      --parent-checkpoint "${checkpoint_path}" \
      --finetune-mode none \
      --frozen \
      --override \
        dataset.train_pack=null \
        dataset.splits.train=null \
        dataset.val_pack=null \
        dataset.splits.val=null "$@"

    polyp_eval_dir="${base_dir}/eval_${POLYP_STEM}"
    python -m ssl4polyp.classification.train_classification \
      --exp-config "${EXP_CONFIG}" \
      --model-key "${model}" \
      --seed "${seed}" \
      --roots "${ROOTS}" \
      --output-dir "${polyp_eval_dir}" \
      --parent-checkpoint "${checkpoint_path}" \
      --finetune-mode none \
      --frozen \
      --override \
        dataset.name="${POLYP_NAME}" \
        dataset.pack="${POLYP_PACK}" \
        dataset.test_pack="${POLYP_PACK}" \
        dataset.splits.test="${POLYP_SPLIT}" \
        dataset.train_pack=null \
        dataset.splits.train=null \
        dataset.val_pack=null \
        dataset.splits.val=null "$@"
  done
done
