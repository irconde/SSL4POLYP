"""Per-frame output CSVs (spec v3.1, section 1.3): schema, loader and unit helpers."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "FRAME_OUTPUT_COLUMNS",
    "FrameOutputs",
    "load_frame_outputs",
    "unit_case",
    "unit_center",
    "unit_e4",
]

# Column order shared by every experiment. Missing values are written as empty strings.
#   case_id         SUN case (virtual 33-frame case for long negative videos); empty for PolypGen
#   sequence_id     source video/sequence (SUN source case, PolypGen negative sequence); empty otherwise
#   center_id       acquisition centre (PolypGen C1..C6); empty when unknown (SUN, PolypGen sequences)
#   origin          pack store id (``sun``, ``polypgen_clean``)
#   morphology      SUN positive morphology (flat / polypoid); empty otherwise
#   negative_source ``sequence`` (negative frame from a video sequence), ``isolated`` (single
#                   negative frame) or ``NA`` (positive frame)
FRAME_OUTPUT_COLUMNS: Tuple[str, ...] = (
    "frame_id",
    "prob",
    "label",
    "pred",
    "case_id",
    "sequence_id",
    "center_id",
    "origin",
    "morphology",
    "negative_source",
)

Unit = Tuple[str, str]  # (stratum, unit name)


@dataclass
class FrameOutputs:
    path: Path
    frame_ids: List[str]
    probs: np.ndarray
    labels: np.ndarray
    preds: np.ndarray
    columns: Dict[str, List[str]] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.labels.size)

    def col(self, name: str) -> List[str]:
        return self.columns.get(name, [""] * len(self))


def load_frame_outputs(path: Path) -> FrameOutputs:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        header = list(reader.fieldnames or [])
    columns = {name: [(row.get(name) or "").strip() for row in rows] for name in header}
    return FrameOutputs(
        path=path,
        frame_ids=columns.get("frame_id", []),
        probs=np.asarray([float(v) for v in columns.get("prob", [])], dtype=np.float64),
        labels=np.asarray([int(v) for v in columns.get("label", [])], dtype=np.int64),
        preds=np.asarray([int(v) for v in columns.get("pred", [])], dtype=np.int64),
        columns=columns,
    )


def unit_case(outputs: FrameOutputs, i: int) -> Optional[Unit]:
    """E1/E2 (SUN): unit = case, stratified by class."""

    return (f"y{int(outputs.labels[i])}", outputs.col("case_id")[i])


def unit_center(outputs: FrameOutputs, i: int) -> Optional[Unit]:
    """E3: unit = centre, stratified by class."""

    return (f"y{int(outputs.labels[i])}", outputs.col("center_id")[i] or "NA")


def unit_e4(outputs: FrameOutputs, i: int) -> Optional[Unit]:
    """E4: positives by centre; negatives by sequence (by centre when there is no sequence)."""

    if int(outputs.labels[i]) == 1:
        return ("pos", outputs.col("center_id")[i] or "NA")
    sequence = outputs.col("sequence_id")[i]
    if sequence:
        return ("neg", f"seq:{sequence}")
    return ("neg", f"center:{outputs.col('center_id')[i] or 'NA'}")


UnitFn = Callable[[FrameOutputs, int], Optional[Unit]]
