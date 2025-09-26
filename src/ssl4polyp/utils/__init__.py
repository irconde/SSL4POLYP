"""Utility helpers for SSL4POLYP."""

from .tensorboard import SummaryWriter

__all__ = [
    "SummaryWriter",
    "enable_deterministic_algorithms",
    "get_MAE_backbone",
    "get_ImageNet_or_random_ViT",
]


def enable_deterministic_algorithms(*, warn_only: bool = True) -> None:
    """Enable deterministic algorithms with compatibility across torch releases.

    Some versions of :func:`torch.use_deterministic_algorithms` do not support the
    ``warn_only`` keyword argument.  The helper tries the modern signature first
    and gracefully falls back to the older API when necessary.
    """

    import torch

    try:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def get_MAE_backbone(weight_path, head, num_classes, frozen, dense, out_token="cls"):

    from ssl4polyp.models import models

    return models.ViT_from_MAE(
        weight_path,
        head,
        num_classes,
        frozen,
        dense,
        embed_dim=768,
        depth=12,
        num_heads=12,
        out_token=out_token,
    )


def get_ImageNet_or_random_ViT(
    head,
    num_classes,
    frozen,
    dense,
    ImageNet_weights,
    out_token="cls",
):

    from ssl4polyp.models import models

    return models.VisionTransformer_from_Any(
        head,
        num_classes,
        frozen,
        dense,
        768,
        12,
        12,
        out_token,
        ImageNet_weights,
    )
