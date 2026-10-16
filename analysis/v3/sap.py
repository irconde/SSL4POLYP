"""Constants fixed by SAP v3.2 revision c (registered in ``results_v3/RUNLOG.md``)."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Dict

REPO = Path(__file__).resolve().parents[2]

SAP_PATH = "docs/v3/plan_analisis_estadistico_v3_2.md"
SAP_SHA256 = "a387aed06f6901affca6c523a4f9ea3b1fca9b34f9e1c7ec69be3ebd36874d47"  # v3.2 revision c

# Section 1
REGIMES = ("sup_imnet", "ssl_imnet", "ssl_colon", "ssl_imnet_matched")
MATCHED = "ssl_imnet_matched"
LABEL = {
    "sup_imnet": "SUP-imnet",
    "ssl_imnet": "SSL-imnet",
    "ssl_colon": "SSL-colon",
    "ssl_imnet_matched": "SSL-imnet-matched",
}
PARTITIONS = (1, 2, 3, 4, 5)
MIN_PARTITIONS = 3
SEEDS = (13, 29, 47)
E1_PERCENTS = (5, 10, 25, 50, 100)
E4_BUDGETS = (50, 100, 200, 500)
E4_ARMS = ("A", "B", "C", "D")
ARM_LABEL = {
    "A": "A: unadapted, inherited τ",
    "B": "B: unadapted, recalibrated τ",
    "C": "C: adapted, inherited τ",
    "D": "D: adapted, recalibrated τ",
}
NEGSOURCE_BUDGET = 50  # H5 / Table 7: 50-frame evaluation split

# Section 2
METRICS = ("auroc", "auprc", "f1", "precision", "recall", "specificity", "balanced_accuracy")
THRESHOLD_FREE = ("auroc", "auprc", "brier", "ece")
CALIBRATION = ("brier", "ece")
PER_POLYP = ("per_polyp_sensitivity_ge1", "per_polyp_sensitivity_ge5")

# Sections 3-5
ALPHA = 0.05
B_BOOT = 2000
BOOT_SEED = 12345
HYPOTHESES = ("H1", "H2", "H3", "H4", "H5", "H6", "H7")
HYPOTHESIS_TEXT: Dict[str, str] = {
    "H1": "E1, SSL-colon − SUP-imnet, AULC of AUPRC (5–100 %)",
    "H2": "E1, SSL-colon − SSL-imnet-matched, AULC of AUPRC (5–100 %)",
    "H3": "E2, SSL-colon − SUP-imnet, AUROC, flat stratum",
    "H4": "E3, SSL-colon − SUP-imnet, AUROC, polypgen_clean_test",
    "H5": "E3 decomposition, sequence − isolated negatives, AUROC, mean of regimes, 50-frame split, common positives",
    "H6": "E4, arm B − arm A, balanced accuracy, mean of budgets and regimes",
    "H7": "E4, arm C − arm A, balanced accuracy, mean of budgets and regimes",
}

# Comparators for SSL-colon in the secondary analyses (section 6)
COMPARATORS = ("sup_imnet", "ssl_imnet", "ssl_imnet_matched")
# E4 arm contrasts reported per budget (section 6: "los cuatro brazos ... incluida D − C")
ARM_CONTRASTS = (("B", "A"), ("C", "A"), ("D", "A"), ("D", "C"), ("D", "B"), ("C", "B"))


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sap_status(repo: Path = REPO) -> Dict[str, object]:
    """Hash check of the SAP this code implements against the file and the last RUNLOG SAP row."""

    file_sha = sha256(repo / SAP_PATH)
    runlog = (repo / "results_v3/RUNLOG.md").read_text(encoding="utf-8")
    rows = re.findall(r"^\| (SAP[^|]*)\| `([^`]+)` \| `([0-9a-f]{64})`", runlog, re.M)
    last = rows[-1] if rows else ("", "", "")
    return {
        "implemented": SAP_SHA256,
        "file_sha256": file_sha,
        "runlog_last": {"name": last[0].strip(), "path": last[1], "sha256": last[2]},
        "file_matches": file_sha == SAP_SHA256,
        "runlog_matches": last[2] == SAP_SHA256,
    }
