from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import torch


def write_outputs(logits: torch.Tensor, metadata: Sequence[Mapping[str, object]], out_dir: Path, tau: float) -> None:
    """Persist evaluation artefacts to ``out_dir``.

    Parameters
    ----------
    logits:
        Tensor of model logits with shape ``(N, C)``.
    metadata:
        Sequence of dictionaries containing metadata for each sample.
    out_dir:
        Directory where outputs will be written. It will be created if missing.
    tau:
        Decision threshold to be saved in ``tau.json``.
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(logits, out_dir / "logits.pt")

    with open(out_dir / "metadata.jsonl", "w") as f:
        for m in metadata:
            json.dump(m, f)
            f.write("\n")

    with open(out_dir / "tau.json", "w") as f:
        json.dump({"tau": tau}, f, indent=2)
