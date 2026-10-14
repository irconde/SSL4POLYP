#!/usr/bin/env python3
"""Compare v3 smoke runs with the equivalent v2 runs (same model, seed, pack).

Both sides are recomputed from the per-frame CSVs at the τ stored in the metrics JSON with the
single metrics module. Rows: sun_full test, E3 (polypgen_clean_test) and E4 s50 arms A
(unadapted, inherited τ) and C (adapted, inherited τ); arms B and D have no v2 counterpart and are
listed with v3 values only. Differences |ΔAUROC| or |ΔAP| above ``--flag`` are flagged.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from ssl4polyp.classification.analysis.frame_outputs import load_frame_outputs
from ssl4polyp.classification.metrics.binary import metrics_at

REPO = Path(__file__).resolve().parents[2]
V2 = REPO / "results" / "reporting_inputs"
TAG = {"sup_imnet": "SUPImNet", "ssl_imnet": "SSLImNet", "ssl_colon": "SSLColon"}
SUN_V2_DIR = {
    "sup_imnet": "exp1_sun_baselines_sup_vs_ssl",
    "ssl_imnet": "exp1_sun_baselines_sup_vs_ssl",
    "ssl_colon": "exp2_sun_baselines_sslcolon_vs_sslimnet",
}
METRICS = (("auroc", "AUROC"), ("auprc", "AP"), ("balanced_accuracy", "BA"), ("recall", "Sens"), ("specificity", "Spec"))


def at(csv_path: Path, tau: float) -> dict:
    out = load_frame_outputs(csv_path)
    return metrics_at(out.probs, out.labels, tau)


def primary_tau(payload: dict) -> float:
    return float(((payload.get("thresholds") or {}).get("primary") or {}).get("tau", payload["test_primary"]["tau"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--v3-root", type=Path, default=REPO / "checkpoints_v3/smoke2/classification/k1")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--flag", type=float, default=0.02)
    parser.add_argument("--out", type=Path, default=REPO / "results_v3/smoke2/k1/comparison_v2.csv")
    args = parser.parse_args()
    s = args.seed
    rows = []
    for model, tag in TAG.items():
        sun3 = json.loads(next((args.v3_root / f"sun_full_{model}_seed{s}").rglob(f"{tag}_SUNFull_s{s}_last.metrics.json")).read_text())
        sun3_csv = next((args.v3_root / f"sun_full_{model}_seed{s}").rglob(f"{tag}_SUNFull_s{s}_test_outputs.csv"))
        sun2_json = V2 / SUN_V2_DIR[model] / f"{tag}_SUNFull_s{s}_last.metrics.json"
        sun2 = json.loads(sun2_json.read_text())
        e3_3_json = next((args.v3_root / f"e3_{model}_seed{s}").rglob(f"{tag}_PolypGenClean_s{s}.metrics.json"))
        e3_3 = json.loads(e3_3_json.read_text())
        e3_2 = json.loads((V2 / "exp5a_polypgen" / f"{tag}_PolypGenClean_s{s}.metrics.json").read_text())
        e4_3_json = next((args.v3_root / f"e4_s50_{model}_seed{s}").rglob(f"{tag}_PolypGenFew50_s{s}_last.metrics.json"))
        e4_3 = json.loads(e4_3_json.read_text())
        e4_2 = json.loads((V2 / "exp5c_polypgen_fewshot" / f"{tag}_PolypGenFew50_s{s}_last.metrics.json").read_text())
        e4_2_zs_tau = float((e4_2.get("zero_shot") or {}).get("test_primary", {}).get("tau", primary_tau(e4_2)))
        arms = e4_3["e4_arms"]
        cases = [
            ("sun_full", sun3_csv, sun3["test_primary"]["tau"],
             V2 / SUN_V2_DIR[model] / f"{tag}_SUNFull_s{s}_test_outputs.csv", float(sun2["test_primary"]["tau"]),
             f"best e{sun3['evaluated_epoch']} (last e{sun3['epoch']}) vs v2 last-epoch weights e{sun2['epoch']}"),
            ("E3", e3_3_json.with_name(f"{tag}_PolypGenClean_s{s}_test_outputs.csv"), e3_3["test_primary"]["tau"],
             V2 / "exp5a_polypgen" / f"{tag}_PolypGenClean_s{s}_test_outputs.csv", float(e3_2["test_primary"]["tau"]), "parent best epoch"),
            ("E4 s50 A", e4_3_json.with_name(f"{tag}_PolypGenFew50_s{s}_zeroshot_outputs.csv"), arms["A"]["metrics"]["tau"],
             V2 / "exp5c_polypgen_fewshot" / f"{tag}_PolypGenFew50_s{s}_zeroshot_outputs.csv", e4_2_zs_tau, "unadapted"),
            ("E4 s50 C", e4_3_json.with_name(f"{tag}_PolypGenFew50_s{s}_test_outputs.csv"), arms["C"]["metrics"]["tau"],
             V2 / "exp5c_polypgen_fewshot" / f"{tag}_PolypGenFew50_s{s}_test_outputs.csv", float(e4_2["test_primary"]["tau"]),
             f"adapted best e{e4_3['evaluated_epoch']}"),
        ]
        for exp, c3, t3, c2, t2, note in cases:
            m3, m2 = at(c3, float(t3)), at(c2, t2)
            row = {"model": model, "experiment": exp, "tau_v3": round(float(t3), 6), "tau_v2": round(t2, 6), "note": note}
            flagged = []
            for key, label in METRICS:
                row[f"{label}_v3"] = m3[key]
                row[f"{label}_v2"] = m2[key]
                row[f"{label}_diff"] = m3[key] - m2[key]
                if label in ("AUROC", "AP") and abs(m3[key] - m2[key]) > args.flag:
                    flagged.append(label)
            row["flag"] = ",".join(flagged)
            rows.append(row)
        for arm in ("B", "D"):
            m = arms[arm]["metrics"]
            row = {"model": model, "experiment": f"E4 s50 {arm}", "tau_v3": round(m["tau"], 6), "tau_v2": "", "note": "no v2 counterpart"}
            for key, label in METRICS:
                row[f"{label}_v3"] = m.get(key, float("nan"))
            rows.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        cells = " ".join(
            f"{label}={r[f'{label}_v3']:.3f}" + (f"/{r[f'{label}_v2']:.3f}({r[f'{label}_diff']:+.3f})" if f"{label}_v2" in r else "")
            for _, label in METRICS
        )
        print(f"{r['model']:9} {r['experiment']:9} {cells} {('FLAG ' + r['flag']) if r.get('flag') else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
