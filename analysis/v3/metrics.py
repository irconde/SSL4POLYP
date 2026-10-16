"""All metrics come from ``src/ssl4polyp/classification/metrics/binary.py`` (spec v3.1 1.4 and 1.10).

``binary.py`` needs only numpy and scikit-learn, but importing it through the package runs
``ssl4polyp.classification.metrics.__init__``, which imports torch. The analysis environment has no
torch, so the module is loaded from its file; the code that runs is the same file the campaign used.
"""
from __future__ import annotations

import importlib.util
import sys
from typing import Dict, Optional

import numpy as np

from .sap import REPO

BINARY_PATH = REPO / "src/ssl4polyp/classification/metrics/binary.py"


def _load_binary():
    name = "ssl4polyp_binary_metrics"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, BINARY_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


binary = _load_binary()


def evaluate(
    probs: np.ndarray,
    labels: np.ndarray,
    tau: float,
    rows: Optional[np.ndarray] = None,
    case_codes: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Binary metrics at ``tau``, Brier, ECE (15 equal-mass bins) and, with ``case_codes``, per-polyp sensitivity.

    ``case_codes`` are per-row case identifiers already restricted to ``rows`` (in a bootstrap
    replicate each drawn copy of a case carries its own code so duplicates count as distinct cases).
    """

    if rows is not None:
        probs, labels = probs[rows], labels[rows]
    out = dict(binary.metrics_at(probs, labels, tau))
    out["brier"] = binary.brier_score(probs, labels)
    out["ece"] = binary.expected_calibration_error(probs, labels, binary.ECE_BINS)
    out["degenerate"] = float(binary.is_degenerate(out))
    if case_codes is not None:
        out.update(binary.per_polyp_sensitivity(probs, labels, case_codes, tau))
    return out
