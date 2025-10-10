from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

pytest.importorskip("numpy")

from ssl4polyp.classification.analysis import exp1_report


def _make_run(*, seed: int, primary: float, baseline: bool = False) -> exp1_report.Exp1Run:
    return exp1_report.Exp1Run(
        model="ssl_imnet" if not baseline else "sup_imnet",
        seed=seed,
        primary_metrics={"f1": primary},
        sensitivity_metrics={"f1": primary},
        tau_primary=0.5,
        tau_sensitivity=0.4,
        frames=tuple(),
        cases={},
        metrics_path=Path(f"seed{seed}.metrics.json"),
        curves={},
        provenance={},
    )


def test_compute_delta_summaries_reports_sample_std() -> None:
    treatment_runs = {
        13: _make_run(seed=13, primary=0.8),
        29: _make_run(seed=29, primary=0.7),
    }
    baseline_runs = {
        13: _make_run(seed=13, primary=0.5, baseline=True),
        29: _make_run(seed=29, primary=0.6, baseline=True),
    }

    summaries = exp1_report._compute_delta_summaries(
        treatment_runs,
        baseline_runs,
        metrics=("f1",),
        bootstrap=0,
        rng_seed=None,
        block="primary",
    )

    summary = summaries.get("f1")
    assert summary is not None
    assert math.isfinite(summary.std)
    expected_std = math.sqrt(0.02)
    assert summary.std == pytest.approx(expected_std)
