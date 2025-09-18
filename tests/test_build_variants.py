import csv
import math
import random
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
import numpy as np

from ssl4polyp.polypdb.robustness.build_variants import (
    add_occlusion,
    adjust_brightness,
    adjust_contrast,
    build_sun_test_corruptions,
    make_blur,
    make_jpeg,
)


def test_adjust_brightness_and_contrast(tmp_path):
    src = tmp_path / "img.png"
    img = np.full((2, 2, 3), 100, dtype=np.uint8)
    cv2.imwrite(str(src), img)
    dst = tmp_path / "bright.png"
    adjust_brightness(src, dst, 0.5)
    out = cv2.imread(str(dst))
    assert np.all(out == 50)

    img2 = np.array([[100, 150], [150, 100]], dtype=np.uint8)
    img2 = np.stack([img2] * 3, axis=-1)
    cv2.imwrite(str(src), img2)
    dst2 = tmp_path / "contrast.png"
    adjust_contrast(src, dst2, 2.0)
    out2 = cv2.imread(str(dst2))
    expected = np.array([[72, 172], [172, 72]], dtype=np.uint8)
    expected = np.stack([expected] * 3, axis=-1)
    assert np.array_equal(out2, expected)


def test_make_blur_and_jpeg(tmp_path):
    src = tmp_path / "img.png"
    img = np.zeros((5, 5, 3), dtype=np.uint8)
    img[2, 2] = 255
    cv2.imwrite(str(src), img)

    dst_blur = tmp_path / "blur.jpg"
    make_blur(src, dst_blur, 1.0)
    out = cv2.imread(str(dst_blur))
    assert out[2, 2, 0] < 255

    dst_jpeg = tmp_path / "out.jpg"
    make_jpeg(src, dst_jpeg, 50)
    assert dst_jpeg.exists()


def test_add_occlusion(tmp_path):
    src = tmp_path / "img.png"
    img = np.full((4, 4, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(src), img)
    dst = tmp_path / "occ.png"
    rng = random.Random(0)
    add_occlusion(src, dst, 0.25, rng)
    out = cv2.imread(str(dst))
    rng2 = random.Random(0)
    size = int(round(math.sqrt(0.25 * 4 * 4)))
    x0 = rng2.randint(0, 2)
    y0 = rng2.randint(0, 2)
    expected = np.full((4, 4, 3), 255, dtype=np.uint8)
    expected[y0 : y0 + size, x0 : x0 + size] = 0
    assert np.array_equal(out, expected)


def test_build_sun_test_corruptions(tmp_path):
    # create minimal dataset
    root_dir = tmp_path / "root"
    root_dir.mkdir()
    img = np.full((4, 4, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(root_dir / "img.png"), img)

    sun_pack = tmp_path / "sun_full"
    sun_pack.mkdir()
    with open(sun_pack / "test.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_path"])
        writer.writerow(["root/img.png"])

    spec = {"occlusion": {"area_ratio": [0.25]}}
    roots = {"root": str(root_dir)}
    out_dir = tmp_path / "out"
    build_sun_test_corruptions(sun_pack, spec, roots, out_dir)

    manifest = out_dir / "manifest.yaml"
    assert manifest.exists()

    variant_dir = out_dir / "occlusion_0.25"
    img_out = variant_dir / "root" / "img.png"
    assert img_out.exists()

    with open(variant_dir / "test.csv") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["variant"] == "occlusion"
    assert rows[0]["severity"] == "0.25"

    rng = random.Random(12345)
    x0 = rng.randint(0, 2)
    y0 = rng.randint(0, 2)
    out_img = cv2.imread(str(img_out))
    assert np.all(out_img[y0 : y0 + 2, x0 : x0 + 2] == 0)
