"""Fixtures for the analysis tests (simulated data only; run in ``ssl4polyp-analysis``).

    PYTHONPATH=. python -m pytest analysis/v3/tests

Monte Carlo sizes can be changed with ``SIM_NULL_REPS`` (default 20), ``SIM_EFFECT_REPS`` (6),
``SIM_NOPART_REPS`` (4), ``SIM_B`` (bootstrap replicates per repetition, 200) and ``SIM_WORKERS`` (16).
``SIM_RESULTS_DIR`` keeps the per-repetition results and their summary.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")
REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from analysis.v3 import simulate as S  # noqa: E402
from analysis.v3 import validate as V  # noqa: E402

WORKERS = int(os.environ.get("SIM_WORKERS", "16"))
B_MC = int(os.environ.get("SIM_B", "200"))


@pytest.fixture(scope="session")
def sim_full(tmp_path_factory):
    """Full-structure simulated campaign with effects and the matched regime in a separate root."""

    out = tmp_path_factory.mktemp("sim_full")
    truth = S.simulate_campaign(S.effect_scenario(), out, sim_seed=11, minimal=False)
    return out, truth


@pytest.fixture(scope="session")
def sim_no_matched(tmp_path_factory):
    out = tmp_path_factory.mktemp("sim_no_matched")
    truth = S.simulate_campaign(S.effect_scenario(include_matched=False), out, sim_seed=12, minimal=True)
    return out, truth


@pytest.fixture(scope="session")
def monte_carlo(tmp_path_factory):
    """Repeated simulations for the three validation scenarios (see ``validate.py``)."""

    work = tmp_path_factory.mktemp("mc")
    n_null = int(os.environ.get("SIM_NULL_REPS", "20"))
    n_eff = int(os.environ.get("SIM_EFFECT_REPS", "6"))
    n_nop = int(os.environ.get("SIM_NOPART_REPS", "4"))
    frames = []
    for sc, seeds in V.scenarios(list(range(100, 100 + n_null)), list(range(200, 200 + n_eff)), list(range(300, 300 + n_nop))):
        frames.append(V.run_reps(sc, seeds, work, B_MC, WORKERS))
    import pandas as pd

    df = pd.concat(frames, ignore_index=True)
    summary = V.summarize(df)
    summary["settings"] = {"B": B_MC, "null_reps": n_null, "effect_reps": n_eff, "nopart_reps": n_nop}
    keep = os.environ.get("SIM_RESULTS_DIR")
    if keep:
        Path(keep).mkdir(parents=True, exist_ok=True)
        df.to_csv(Path(keep) / "validation_reps.csv", index=False, float_format="%.8g")
        (Path(keep) / "validation_summary.json").write_text(json.dumps(summary, indent=1))
    return df, summary
