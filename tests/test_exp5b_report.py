import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ssl4polyp.classification.analysis import exp5b_report as module  # type: ignore[import]


def test_parse_case_metrics_block_preserves_auxiliary_fields():
    block = {
        "clean": [
            {
                "case_id": "case-001",
                "metrics": {"f1": 0.75, "precision": "0.8", "invalid": None},
                "clusters": {"patient": "subj-1"},
                "metadata": {"camera": "endo-3"},
                "note": "primary",
            },
            {
                "case": "case-002",
                "accuracy": 0.6,
                "clusters": {"patient": "subj-2"},
            },
        ],
        "blur_sigma_1p5": {
            "case-003": {"f1": 0.42},
        },
    }

    parsed = module._parse_case_metrics_block(block)

    clean_cases = parsed["clean"]
    case_one = clean_cases["case-001"]
    case_two = clean_cases["case-002"]

    assert math.isclose(case_one["f1"], 0.75)
    assert math.isclose(case_one["precision"], 0.8)
    assert case_one["clusters"] == {"patient": "subj-1"}
    assert case_one["metadata"] == {"camera": "endo-3"}
    # extra non-metric fields are also preserved
    assert case_one["note"] == "primary"
    # entries without a nested metrics block should retain their original structure
    assert math.isclose(case_two["accuracy"], 0.6)
    assert case_two["clusters"] == {"patient": "subj-2"}

    blur_cases = parsed["blur_sigma_1p5"]
    assert math.isclose(blur_cases["case-003"]["f1"], 0.42)
