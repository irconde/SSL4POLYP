"""Standardised image transforms and perturbation utilities for classification packs."""
from __future__ import annotations

import hmac
import hashlib
import io
import math
import random
from dataclasses import dataclass
from typing import Mapping, MutableMapping, Optional, Sequence

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import torch
from torchvision import transforms as T

IMAGENET_MEAN: Sequence[float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Sequence[float] = (0.229, 0.224, 0.225)
DEFAULT_HMAC_KEY = b"ssl4polyp"


def _truthy(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _token_to_float(token: str) -> Optional[float]:
    token = token.strip().lower()
    if not token:
        return None
    token = token.replace("minus", "-")
    token = token.replace("neg", "-")
    token = token.replace("p", ".")
    try:
        return float(token)
    except ValueError:
        return None


def _parse_sigma(variant: str) -> Optional[float]:
    parts = variant.split("_")
    for part in reversed(parts):
        sigma = _token_to_float(part)
        if sigma is not None:
            return sigma
    return None


def _parse_quality(variant: str) -> Optional[int]:
    value = _parse_sigma(variant)
    if value is None:
        return None
    return int(round(value))


def _parse_bc(variant: str) -> tuple[Optional[float], Optional[float]]:
    brightness: Optional[float] = None
    contrast: Optional[float] = None
    for part in variant.split("_"):
        if part.startswith("b"):
            brightness = _token_to_float(part[1:])
        elif part.startswith("c"):
            contrast = _token_to_float(part[1:])
    return brightness, contrast


def _parse_occ(variant: str) -> Optional[float]:
    if "a" in variant:
        token = variant.split("a", 1)[1]
    else:
        token = variant.split("_")[-1]
    return _token_to_float(token)


def _apply_jpeg(img: Image.Image, quality: int) -> Image.Image:
    quality = max(1, min(int(quality), 100))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality, optimize=False, subsampling=0)
    buffer.seek(0)
    degraded = Image.open(buffer)
    degraded.load()
    return degraded.convert("RGB")


def _apply_brightness_contrast(
    img: Image.Image, brightness: Optional[float], contrast: Optional[float]
) -> Image.Image:
    result = img
    if brightness is not None and brightness > 0:
        result = ImageEnhance.Brightness(result).enhance(brightness)
    if contrast is not None and contrast > 0:
        result = ImageEnhance.Contrast(result).enhance(contrast)
    return result


def _apply_occlusion(img: Image.Image, area_fraction: float, rng: random.Random) -> Image.Image:
    area_fraction = max(0.0, min(float(area_fraction), 1.0))
    if area_fraction <= 0:
        return img
    result = img.copy()
    width, height = result.size
    total_area = width * height
    occ_area = max(1.0, area_fraction * total_area)
    aspect = rng.uniform(0.5, 2.0)
    occ_w = int(round(math.sqrt(occ_area * aspect)))
    occ_h = int(round(math.sqrt(occ_area / aspect)))
    occ_w = max(1, min(width, occ_w))
    occ_h = max(1, min(height, occ_h))
    max_x = max(0, width - occ_w)
    max_y = max(0, height - occ_h)
    x0 = rng.randint(0, max_x) if max_x > 0 else 0
    y0 = rng.randint(0, max_y) if max_y > 0 else 0
    x1 = min(width, x0 + occ_w)
    y1 = min(height, y0 + occ_h)
    draw = ImageDraw.Draw(result)
    draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0))
    return result


def _row_hmac_seed(row: Mapping[str, object], key: bytes) -> int:
    digest = hmac.new(
        key,
        "|".join(
            [
                str(row.get(field, ""))
                for field in (
                    "frame_path",
                    "frame_id",
                    "case_id",
                    "variant",
                    "perturbation_id",
                )
            ]
        ).encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


class PerRowPerturbations:
    """Apply deterministic perturbations based on metadata annotations."""

    def __init__(self, key: bytes = DEFAULT_HMAC_KEY) -> None:
        self.key = key

    def __call__(self, img: Image.Image, row: Mapping[str, object] | None) -> Image.Image:
        if not row:
            return img
        if not _truthy(row.get("render_in_pipeline", True)):
            return img
        variant = str(row.get("variant") or row.get("perturbation_id") or "").strip()
        if not variant or variant.lower() == "clean":
            return img
        rng_seed = row.get("rng_seed")
        seed = int(rng_seed) if rng_seed not in (None, "", -1, "-1") else _row_hmac_seed(row, self.key)
        rng = random.Random(seed)
        variant_lower = variant.lower()
        if variant_lower.startswith("blur"):
            sigma = row.get("blur_sigma")
            sigma_val = float(sigma) if sigma not in (None, "", -1, "-1", "-1.0") else None
            if sigma_val is None:
                sigma_val = _parse_sigma(variant_lower)
            if sigma_val is None or sigma_val <= 0:
                return img
            return img.filter(ImageFilter.GaussianBlur(radius=float(sigma_val)))
        if variant_lower.startswith("jpeg"):
            quality = row.get("jpeg_q")
            quality_val = float(quality) if quality not in (None, "", -1, "-1") else None
            if quality_val is None:
                quality_val = _parse_quality(variant_lower)
            if quality_val is None:
                return img
            return _apply_jpeg(img, int(round(quality_val)))
        if variant_lower.startswith("bc"):
            brightness = row.get("brightness")
            contrast = row.get("contrast")
            brightness_val = (
                float(brightness)
                if brightness not in (None, "", -1, "-1", "-1.0")
                else None
            )
            contrast_val = (
                float(contrast)
                if contrast not in (None, "", -1, "-1", "-1.0")
                else None
            )
            parsed_b, parsed_c = _parse_bc(variant_lower)
            brightness_val = brightness_val if brightness_val is not None else parsed_b
            contrast_val = contrast_val if contrast_val is not None else parsed_c
            return _apply_brightness_contrast(img, brightness_val, contrast_val)
        if variant_lower.startswith("occ"):
            area = row.get("bbox_area_frac")
            area_val = float(area) if area not in (None, "", -1, "-1", "-1.0") else None
            if area_val is None:
                area_val = _parse_occ(variant_lower)
            if area_val is None or area_val <= 0:
                return img
            return _apply_occlusion(img, area_val, rng)
        return img


@dataclass
class ClassificationTransforms:
    """Factory for standard train/val/test transforms with optional perturbations."""

    stage: str
    image_size: int = 224
    mean: Sequence[float] = IMAGENET_MEAN
    std: Sequence[float] = IMAGENET_STD
    enable_augmentations: Optional[bool] = None
    enable_perturbations: bool = False
    hmac_key: bytes = DEFAULT_HMAC_KEY

    def __post_init__(self) -> None:
        stage = self.stage.lower()
        if stage not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported stage '{self.stage}'.")
        self.stage = stage
        if self.enable_augmentations is None:
            self.enable_augmentations = self.stage == "train"
        self._pre_tensor = self._build_pre_tensor()
        self._post_tensor = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=self.mean, std=self.std),
            ]
        )
        self._perturber = PerRowPerturbations(self.hmac_key) if self.enable_perturbations else None

    def _build_pre_tensor(self) -> T.Compose:
        ops: list = [T.Resize((self.image_size, self.image_size))]
        if self.enable_augmentations:
            ops.extend(
                [
                    T.ColorJitter(brightness=0.4, contrast=0.5, saturation=0.25, hue=0.01),
                    T.GaussianBlur((25, 25), sigma=(0.001, 2.0)),
                    T.RandomHorizontalFlip(),
                    T.RandomVerticalFlip(),
                    T.RandomRotation(180),
                ]
            )
        return T.Compose(ops)

    def __call__(self, img: Image.Image, row: Optional[Mapping[str, object]] = None) -> torch.Tensor:
        img = img.convert("RGB")
        img = self._pre_tensor(img)
        if self._perturber is not None:
            img = self._perturber(img, row)
        tensor = self._post_tensor(img)
        return tensor.float()


__all__ = [
    "ClassificationTransforms",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "PerRowPerturbations",
]
