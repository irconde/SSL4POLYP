from typing import Any

import torch.nn as nn

__all__ = [
    "normalise_finetune_mode",
    "configure_finetune_parameters",
]


def normalise_finetune_mode(raw: Any, *, default: str = "full") -> str:
    """Resolve a user-specified fine-tuning regime into a canonical label."""

    valid_modes = {"none", "full", "head+2"}

    if raw is None:
        mode = str(default).strip().lower()
    else:
        mode = str(raw).strip().lower()
    if not mode:
        mode = str(default).strip().lower()

    if mode not in valid_modes:
        raise ValueError(
            f"Unsupported fine-tuning mode '{raw}'. Expected one of {sorted(valid_modes)}."
        )

    return mode


def configure_finetune_parameters(model: nn.Module, mode: str) -> None:
    """Toggle ``requires_grad`` flags according to the desired fine-tuning regime."""

    resolved_mode = normalise_finetune_mode(mode, default="full")

    if resolved_mode == "full":
        for param in model.parameters():
            param.requires_grad_(True)
    else:
        for param in model.parameters():
            param.requires_grad_(False)

        head_modules: list[nn.Module] = []
        lin_head = getattr(model, "lin_head", None)
        if isinstance(lin_head, nn.Module):
            head_modules.append(lin_head)
        head_module = getattr(model, "head", None)
        if isinstance(head_module, nn.Module) and head_module is not lin_head:
            head_modules.append(head_module)

        for head in head_modules:
            for param in head.parameters():
                param.requires_grad_(True)

        if resolved_mode == "head+2":
            blocks = getattr(model, "blocks", None)
            if isinstance(blocks, (nn.ModuleList, list, tuple)) and len(blocks) > 0:
                for block in list(blocks)[-2:]:
                    for param in block.parameters():
                        param.requires_grad_(True)

    if hasattr(model, "frozen"):
        model.frozen = resolved_mode == "none"
