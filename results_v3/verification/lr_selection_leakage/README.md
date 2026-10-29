# Learning-rate selection: case overlap between selection validation and campaign tests

Reviewer question: the registered learning-rate sweep scored a reference validation split that may share cases with
the test splits of the five campaign partitions. This directory quantifies the overlap and checks whether it changes
the selection. Analysis of existing data only: nothing was trained and no registered result was modified.

    PYTHONPATH=. python scripts/v3/lr_selection_leakage.py --out results_v3/verification/lr_selection_leakage

`ssl4polyp-analysis`, run 2026-10-29 (`run_log.txt`).

## Definitions

- **Sweep validation**: `data_packs/sun_full/val.csv` (v2 `sun_full`; 15 positive and 15 negative cases, 990 frames).
  The deviation was declared in `results_v3/SAP_deviations.md` (v3.2): this split shares cases with the new k = 1.
  The 36 sweep CSVs (4 regimes × 3 learning rates × 3 seeds) cover exactly these cases.
- **Campaign tests**: every `test_pack` in `results_v3/verification/campaign_plan_all4.csv`:
  - SUN `v3/k<k>/base` test of k1..k5 (E1, E2);
  - PolypGen clean test (E3);
  - the E4 few-shot tests.

  PolypGen shares no case with SUN, so only the SUN tests contribute to the overlap.
- **Case**: `case_id`. A positive case is a polyp sequence. A negative case is a 33-frame chunk of a long negative
  video (`source_case_id`): every split's 15 negative cases come from only 3 source videos. The overlap is also given
  by frame and, for negatives, by source video.

## Files

| File | Content |
|---|---|
| `overlap_sweep_val_vs_tests.csv` | 1a: sweep validation vs test of k1..k5 and the union of all campaign tests |
| `overlap_partition_val_vs_test.csv` | 1b: validation of k vs test of j (5 × 5), the splits used by the per-budget experiment |
| `sweep_val_ap_runs.csv` | 2: AP for each sweep run on all frames (equals `lr_sweep_runs{,_matched}.csv`, max diff 2.2e-16), without shared cases, and also without negatives from shared source videos |
| `sweep_val_ap_by_lr.csv`, `sweep_selection.csv` | 2: mean AP by regime and learning rate, with the SAP section 10 rule (highest mean) |
| `per_budget_selection_by_partition.csv` | 3: registered rule (gain ≥ 1e-4 over the registered learning rate) within each partition on its 3 seeds, vs the pooled selection |
| `contrasts_partition_selection_long.csv` | 3: SSL-colon contrasts (AULC and per budget) with per-partition selection (`partition`), pooled selection (`pooled`), registered, and the two differences |
| `registered_check.csv`, `summary.json` | checks: registered version = `results_v3/analysis/contrast_summary.csv` (108/108); `pooled` = committed per-budget results (max diff 4.9e-11) |

## Results

**1a.** The sweep validation shares the following cases with the campaign tests:

| | k1 | k2 | k3 | k4 | k5 | union |
|---|---|---|---|---|---|---|
| positive cases (of 15) | 2 | 2 | 6 | 3 | 3 | 9 |
| negative cases (of 15) | 0 | 1 | 0 | 0 | 0 | 1 |
| negative source videos (of 3) | 0 | 1 | 0 | 1 | 1 | 2 |

**1b.** Validation of k and test of j are disjoint when k = j. Off the diagonal, they share 1–6 positive cases and 0–1
negative cases.

**2.** Without the frames of the 10 cases in any campaign test, 6 positive and 14 negative cases remain
(198 + 462 frames). The selected learning rate does not change for any regime.

The stricter variant also removes negatives from the 2 shared source videos. It leaves 6 positive cases but only 3
negative cases from a single video, too few to support a selection. In it:
- SUP-imnet would move to 1e-4: mean AP 0.9943 against 0.9940 at 3e-4.
- SSL-imnet would also move to 1e-4: 0.9990 against 0.9986.

**3.** The per-partition selection differs from the pooled one in 23 of 100 cells. H1 with per-partition selection is
0.0254, against 0.0287 pooled and 0.0475 registered.

Exploratory: no Holm, no verdicts. Registered results are unchanged.
