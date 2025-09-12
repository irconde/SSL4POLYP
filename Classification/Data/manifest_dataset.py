from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, List

from PIL import Image
from torch.utils import data


class ManifestDataset(data.Dataset):
    """Dataset backed by manifest ``rows`` with resolved paths.

    Parameters
    ----------
    paths:
        Sequence of resolved image paths.
    labels:
        Optional sequence of numeric labels.  If ``None`` the dataset will
        return only images (and optionally metadata).
    meta:
        Sequence of rows as returned by :func:`manifests.load_pack`.  Each row
        is a mutable mapping containing at least the ``frame_path`` entry and
        possibly additional metadata such as ``frame_id``.
    transform_input:
        Optional transformation to apply to the loaded PIL image.
    return_meta:
        If ``True`` the dataset will return the metadata dictionary alongside
        the image (and label if available).
    """

    def __init__(
        self,
        paths: Sequence[Path],
        labels: Optional[Sequence[Any]] = None,
        meta: Optional[Sequence[Mapping[str, Any]]] = None,
        transform_input=None,
        return_meta: bool = False,
    ) -> None:
        self.paths: List[str] = [str(p) for p in paths]
        self.labels = list(labels) if labels is not None else None
        self.meta = list(meta) if meta is not None else [{} for _ in paths]
        self.transform_input = transform_input
        self.return_meta = return_meta

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.paths)

    def __getitem__(self, index: int):  # type: ignore[override]
        path = self.paths[index]
        img = Image.open(path).resize((224, 224))
        if self.transform_input is not None:
            img = self.transform_input(img)

        row = self.meta[index] if index < len(self.meta) else {}
        if self.labels is None:
            if self.return_meta:
                return img.float(), row
            return img.float()

        label = self.labels[index]
        if self.return_meta:
            return img.float(), label, row
        return img.float(), label
