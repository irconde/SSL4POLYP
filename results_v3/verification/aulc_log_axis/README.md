# E1 AULC: linear (registered) vs logarithmic budget axis

**Post hoc sensitivity analysis, after the registered analysis and not part of the SAP.** It asks whether the E1 AULC
contrasts depend on integrating over the linear budget axis. Analysis of existing data only: nothing was trained and no
registered result was modified.

    PYTHONPATH=. python scripts/v3/aulc_log_axis.py --out results_v3/verification/aulc_log_axis

`ssl4polyp-analysis`, run 2026-10-30 (`run_log.txt`), registered campaign `checkpoints_v3/campaign/classification`.

## Definitions

- **Linear AULC (registered, SAP section 2):** trapezoid of the per-budget metric over the budgets 5, 10, 25, 50,
  100 %, divided by the axis width (95).
- **Log AULC:** trapezoid over ln(budget), divided by ln(100/5).

Both are weighted sums of the per-budget metrics (`aulc_weights.csv`). The linear axis puts 66 % of the weight on
50–100 %. The log axis spreads it more evenly: 12 % at 5 %, 27 % at 10 %, 27 % at 25 %, 23 % at 50 % and 12 % at 100 %.

## Inference

The registered code (`analysis/v3`) is used unchanged:

- **Estimate:** paired, the mean of the 15 per-run values (5 partitions × 3 seeds).
- **t-test CI:** on the 5 partition means (`mixed`, variant `ttest`).
- **Bootstrap CI:** cluster bootstrap of the e1 family (B = 2000, seed 12345).

The 244 registered e1 contrasts are bootstrapped alongside, so the resampling spaces and the draws are those of the
registered analysis.

## Contrasts

SSL-colon minus each comparator, for AULC of AUPRC and of AUROC:

| Comparator | AUPRC | AUROC |
|---|---|---|
| SUP-imnet | H1 | secondary |
| SSL-imnet | secondary | secondary |
| SSL-imnet-matched | H2 | secondary |

## Check

The linear contrasts are built by the same weights function as the log ones, applied to the budget instead of its
logarithm. They reproduce the registered values (`results_v3/analysis/contrast_summary.csv`): 48/48 quantities equal,
max |diff| 4.3e-12 (`registered_check.csv`). The quantities are estimate, t-test CI, bootstrap CI and B; H1 and H2 are
checked against both their hypothesis and their secondary rows. The script stops if any differs.

## Files

| File | Content |
|---|---|
| `aulc_linear_vs_log.csv`, `.md` | one row per contrast: linear and log estimate, t-test CI, bootstrap CI, log − linear |
| `aulc_axis_long.csv` | one row per contrast and axis, with weights, partition and seed SD, unadjusted p and B |
| `aulc_weights.csv` | weight of each budget on each axis |
| `registered_check.csv`, `summary.json`, `run_log.txt` | reproduction check and run information |

## Results

| Contrast | Linear AULC (registered) | Log AULC |
|---|---|---|
| H1: SSL-colon − SUP-imnet, AUPRC | 0.0475 [0.0286, 0.0663; 0.0355, 0.0592] | 0.0742 [0.0408, 0.1075; 0.0554, 0.0908] |
| SSL-colon − SSL-imnet, AUPRC | 0.0104 [−0.0038, 0.0246; 0.0013, 0.0197] | 0.0422 [0.0156, 0.0688; 0.0280, 0.0561] |
| H2: SSL-colon − SSL-imnet-matched, AUPRC | −0.0138 [−0.0285, 0.0009; −0.0214, −0.0064] | −0.0124 [−0.0332, 0.0084; −0.0237, −0.0011] |
| SSL-colon − SUP-imnet, AUROC | 0.0435 [0.0277, 0.0593; 0.0315, 0.0569] | 0.0578 [0.0340, 0.0817; 0.0408, 0.0762] |
| SSL-colon − SSL-imnet, AUROC | 0.0089 [−0.0031, 0.0209; 0.0001, 0.0186] | 0.0365 [0.0145, 0.0585; 0.0234, 0.0512] |
| SSL-colon − SSL-imnet-matched, AUROC | −0.0150 [−0.0321, 0.0020; −0.0234, −0.0071] | −0.0147 [−0.0406, 0.0111; −0.0270, −0.0017] |

Each cell gives the estimate, then the t-test CI and the bootstrap CI.

- The log axis weights the small budgets more, where SSL-colon's advantage over SUP-imnet and SSL-imnet is largest.
  Registered per-budget AUPRC differences, at 5 % and at 10 %:
  - vs SUP-imnet: +0.120 and +0.138;
  - vs SSL-imnet: +0.166 and +0.100.

  Both contrasts therefore grow.
- Against SSL-imnet, both intervals now lie above zero, for AUPRC and for AUROC.
- The contrasts against SSL-imnet-matched barely change (log − linear: +0.0014 AUPRC, +0.0003 AUROC). They stay
  negative, with t-test CIs that include zero and bootstrap CIs below zero. Against SSL-imnet-matched the per-budget
  AUPRC difference is positive only at 5 % (+0.038) and negative from 10 % to 100 %, so reweighting the budgets
  changes little.

Sensitivity analysis: no Holm, no verdicts. Registered results are unchanged.
