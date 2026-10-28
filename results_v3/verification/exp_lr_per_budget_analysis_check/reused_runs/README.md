# Code check only: not results of the experiment

These files are a test of the analysis code (`analysis/v3/exploratory_lr_per_budget/`), written before any new run of
the per-budget learning-rate experiment existed (plan revision 1, check 3). The selection was restricted to the
registered runs and the 300 reused runs, so for most regimes and budgets not every candidate learning rate was
available. The "selected" and "diff" values therefore come from an incomplete selection and **are not results of the
experiment**; do not quote them. The only purpose of this run was to confirm that the recomputed registered values equal
the registered analysis (`registered_check.csv`). The results of the experiment are in
`results_v3/analysis_exploratory_lr_per_budget/`.
