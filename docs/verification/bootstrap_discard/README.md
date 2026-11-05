# Discarded bootstrap replicates of H5, H6 and H7

Date: 2026-11-05 (commit `c3bae2c`). Post hoc check, not registered; registered results are read, never written.

```bash
PYTHONPATH=. python docs/verification/bootstrap_discard/boot_discard.py --out docs/verification/bootstrap_discard/output
```

Run from the repository root in `ssl4polyp-analysis`, with the registered campaign under
`checkpoints_v3/campaign/classification` (about 8 min with 16 worker processes). Outputs in `output/`: `intervals.csv`,
`packs_500.csv`, `registered_check.csv`, `summary.json`, `run_log.txt`. Two runs gave identical files (except the
elapsed time in `summary.json`).

**Registered values reproduced.** The script uses the registered bootstrap (B = 2000, seed 12345) over the registered
E4 and negative-source spaces (checked). It reproduces the 1,691 valid replicates of H6 and H7, the 1,997 of H5 and
their bootstrap intervals, with difference 0 at the 10 significant digits of `results_v3/analysis/contrast_summary.csv`
(raw differences ≤ 5.1e-12, from the summation order). See `registered_check.csv`.

**Why H6 and H7 lose 309 replicates.** Replicates are discarded only at 500 frames (0 at 50, 100 and 200). The PolypGen
negatives are resampled as 24 units: 18 sequences and 6 centres with isolated negatives, drawn with replacement. The
500-frame evaluation split of each pack holds only sequence negatives, from 4 of the 18 sequences. A replicate is
undefined for a pack when none of its 4 sequences is drawn, with probability (20/24)^24 = 0.0126 per pack (18 to 35
replicates per pack, `packs_500.csv`). It is discarded when this happens to any of the 15 packs. The exact probability,
by inclusion–exclusion over the packs' sequence sets, is 0.157, i.e. 314 expected discards; 309 observed.

**Why H5 loses 3 replicates.** H5 compares sequence and isolated negatives. The isolated view is undefined when none of
the 6 centres with isolated negatives is drawn, with probability (18/24)^24 = 0.0010, i.e. 2.0 expected discards; 3
observed. A replicate without sequence negatives has probability 3.6e-15.

**Intervals with and without discarding** (balanced accuracy, bootstrap 95 % interval, `intervals.csv`). The alternative
averages, in each run, the budgets that are defined in that replicate, so no replicate is discarded:

| Hypothesis | Registered (discard) | Average of defined budgets | Largest change of a limit |
|---|---|---|---|
| H6 (B − A) | [−0.01838, 0.00578], B = 1,691 | [−0.01864, 0.00614], B = 2,000 | 0.0004 |
| H7 (C − A) | [−0.13529, −0.01499], B = 1,691 | [−0.13733, −0.01139], B = 2,000 | 0.0036 |
| H5 (sequence − isolated, AUROC) | [−0.03101, 0.30935], B = 1,997 | not applicable (one budget) | — |
