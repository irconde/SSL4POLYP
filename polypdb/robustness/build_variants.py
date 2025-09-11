from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import yaml


def make_blur(src: Path, dst: Path, sigma: float) -> None:
    """Apply Gaussian blur with sigma and deterministic kernel size."""
    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    k = max(3, int(2 * round(3 * sigma) + 1))
    blurred = cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), blurred)


def make_jpeg(src: Path, dst: Path, quality: int) -> None:
    """Re-encode the image with the given JPEG quality."""
    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])


def adjust_brightness(src: Path, dst: Path, factor: float) -> None:
    """Scale image brightness by ``factor``."""
    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED).astype(np.float32)
    adjusted = np.clip(img * factor, 0, 255).astype(np.uint8)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), adjusted)


def adjust_contrast(src: Path, dst: Path, factor: float) -> None:
    """Adjust image contrast around mid-gray (128)."""
    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED).astype(np.float32)
    adjusted = np.clip((img - 128.0) * factor + 128.0, 0, 255).astype(np.uint8)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), adjusted)


def add_occlusion(src: Path, dst: Path, area_ratio: float, rng: random.Random) -> None:
    """Add a black square occlusion covering ``area_ratio`` of the image."""
    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    h, w = img.shape[:2]
    size = max(1, int(round(math.sqrt(area_ratio * h * w))))
    size = min(size, h, w)
    x_max = w - size
    y_max = h - size
    x0 = rng.randint(0, x_max) if x_max > 0 else 0
    y0 = rng.randint(0, y_max) if y_max > 0 else 0
    img[y0:y0 + size, x0:x0 + size] = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), img)


def build_sun_test_corruptions(
    sun_full_pack: Path,
    spec: Dict,
    roots: Dict[str, str],
    out_dir: Path,
) -> None:
    """Build corrupted SUN test variants according to ``spec``.

    Parameters
    ----------
    sun_full_pack:
        Path containing ``test.csv`` describing the SUN test set.
    spec:
        Mapping of corruption name to parameter list.
    roots:
        Mapping of root identifiers to filesystem paths.
    out_dir:
        Output directory for corrupted copies and manifests.
    """

    test_csv = sun_full_pack / "test.csv"
    with open(test_csv, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    seed = 12345
    out_dir.mkdir(parents=True, exist_ok=True)

    for variant, params in spec.items():
        # Expect a single parameter per variant
        param_name, values = next(iter(params.items()))
        for idx, value in enumerate(values):
            variant_dir = out_dir / f"{variant}_{value}"
            variant_dir.mkdir(parents=True, exist_ok=True)
            rng = random.Random(seed + idx)
            out_rows = []
            for row in rows:
                path = Path(row["frame_path"])
                root = path.parts[0]
                rel = Path(*path.parts[1:])
                src = Path(roots[root]) / rel
                dst = variant_dir / root / rel

                if variant == "blur":
                    make_blur(src, dst, float(value))
                elif variant == "jpeg":
                    make_jpeg(src, dst, int(value))
                elif variant == "brightness":
                    adjust_brightness(src, dst, float(value))
                elif variant == "contrast":
                    adjust_contrast(src, dst, float(value))
                elif variant == "occlusion":
                    add_occlusion(src, dst, float(value), rng)
                else:
                    raise ValueError(f"Unknown variant: {variant}")

                new_row = dict(row)
                new_row["frame_path"] = str(Path(root) / rel)
                new_row["variant"] = variant
                new_row["severity"] = str(value)
                out_rows.append(new_row)

            with open(variant_dir / "test.csv", "w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=fieldnames + ["variant", "severity"]
                )
                writer.writeheader()
                writer.writerows(out_rows)

    manifest = {"spec": spec, "policy": {"robustness_rng": seed}}
    with open(out_dir / "manifest.yaml", "w") as f:
        yaml.safe_dump(manifest, f)
