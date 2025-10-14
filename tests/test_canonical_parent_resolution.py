from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from ssl4polyp.classification.train_classification import (
    _resolve_canonical_sun_parent_checkpoint,
)


@pytest.mark.parametrize(
    "model_key, seed, expected_stem",
    [
        ("sup_imnet", 13, "SUPImNet__SUNFull_s13.pth"),
        ("ssl_imnet", 29, "SSLImNet__SUNFull_s29.pth"),
        ("ssl_colon", 47, "SSLColon__SUNFull_s47.pth"),
    ],
)
def test_resolve_canonical_parent_checkpoint(tmp_path: Path, model_key: str, seed: int, expected_stem: str) -> None:
    output_dir = tmp_path / "classification" / f"exp5a_{model_key}_seed{seed}"
    args = Namespace(output_dir=str(output_dir))
    selected_model = {"key": model_key}
    parent_path = _resolve_canonical_sun_parent_checkpoint(
        args,
        selected_model,
        model_key,
        seed,
    )
    experiment_prefix = {
        "sup_imnet": "exp1_sup_imnet",
        "ssl_imnet": "exp1_ssl_imnet",
        "ssl_colon": "exp2_ssl_colon",
    }[model_key]
    expected = (
        tmp_path
        / "classification"
        / f"{experiment_prefix}_seed{seed}"
        / "sun_baselines"
        / expected_stem
    )
    assert parent_path == expected


def test_resolve_canonical_parent_unsupported_model(tmp_path: Path) -> None:
    args = Namespace(output_dir=str(tmp_path / "classification" / "exp5a_unknown_seed7"))
    with pytest.raises(KeyError):
        _resolve_canonical_sun_parent_checkpoint(args, {"key": "unknown"}, "unknown", 7)
