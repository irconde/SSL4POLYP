"""Utility helpers exposed under :mod:`ssl4polyp.utils`."""

from ssl4polyp.models import models

__all__ = [
    "get_MAE_backbone",
    "get_ImageNet_or_random_ViT",
]


def get_MAE_backbone(weight_path, head, num_classes, frozen, dense, out_token="cls"):
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
