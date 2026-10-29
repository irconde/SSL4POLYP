# SSL-colon-r analysis code: identity check (not results)

Output of `analysis/v3/exploratory_sslcolon_r.py` run before SSL-colon-r exists, with the original SSL-colon substituted
for itself (plan `docs/v3/exploratory_sslcolon_repretrain_plan.md`):

    PYTHONPATH=. python -m analysis.v3.exploratory_sslcolon_r --out <scratch> \
        --r-root checkpoints_v3/campaign/classification --r-regime ssl_colon \
        --loss-r checkpoints_v3/pretrain/mae_imnet_matched/ckpts/log.txt

- `registered_check.csv`: versions `r` and `registered` against `results_v3/analysis/contrast_summary.csv`, 84/84 equal
  (estimate, t-test CI, bootstrap CI, B; max |diff| 4.7e-11 at its 10 stored significant digits).
- `hypotheses_long.csv`: `diff` = r − registered is 0 up to rounding (|diff| ≤ 3e-17).
- `pretraining_loss_r_vs_matched.csv`: the SSL-imnet-matched log compared with itself (400 epochs, differences 0); a
  check of the loss-curve code only.

These files are a code check, not results of the experiment. `run_metrics_ssl_colon_and_r.csv` (per-run metrics of the
registered SSL-colon, duplicated) was not kept.
