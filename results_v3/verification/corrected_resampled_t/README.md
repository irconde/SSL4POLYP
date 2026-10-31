# H1–H7 with the corrected resampled t-test (Nadeau–Bengio)

**Post hoc sensitivity analysis, after the registered analysis and not part of the SAP.** A reviewer notes that the five
SUN partitions share test and training cases, so their means are not independent and the t-test on partition means
underestimates the variance. This analysis:

- reports whether the calibration simulation reproduced that structure;
- recomputes H1–H7 with the corrected resampled t-test;
- counts the overlap between partitions.

Analysis of existing data only: nothing was trained and no registered result was modified.

    PYTHONPATH=. python scripts/v3/corrected_resampled_t.py --out results_v3/verification/corrected_resampled_t

`ssl4polyp-analysis`, run 2026-10-31 (`run_log.txt`). Inputs: the registered per-run differences
(`results_v3/analysis/paired_differences.csv`), the registered summaries, and the packs `data_packs/v3/k*/`.

## 1. Did the calibration simulation reuse the real partition structure?

**No.** The simulated campaigns of `results_v3/verification/analysis_v3_variants/` (and `analysis_v3_validation/`)
generate synthetic SUN partitions, independently of the real ones (`analysis/v3/simulate.py`).

- **Synthetic pool:** `sun_pools` (line 199) builds 100 positive cases `SUN_0001…SUN_0100` and 100 negative virtual
  cases from 13 synthetic videos. These are not the real cases.
- **Partitions:** `partition_splits` (line 221) draws each partition's test and validation cases with its own
  generator, `np.random.default_rng(9000 + k)` (line 222), by independent permutations of the pool. Test sets therefore
  overlap only by chance, not as the real partitions do. With these seeds, pairs of simulated test sets share 1–3
  positive cases, 1–5 negative virtual cases and 4–9 negative videos.
- **No training:** nothing is trained in the simulation, so the overlap of the training sets is not represented at
  all.
- **Partition effects:** the variation between partitions is imposed as independent random effects. The docstring
  (lines 16–23) says that each run's scores are solved so that the hypothesis metric on the realised test population
  equals

      target(treatment) = target(reference) + effect + v_k + w_{k,s},   v_k ~ N(0, sd_partition), w_{k,s} ~ N(0, sd_seed)

  and `Generator.re` (lines 310–315) draws one `v_k` per partition, independently:

      v = {k: self.rng.normal(0, self.sc.sd_partition[h]) if self.sc.sd_partition[h] > 0 else 0.0 for k in self.sc.partitions}

  The case effects (`case_eff`, line 348) are shared by all models and cancel in the per-run differences, which are
  fixed by the targets.

The partition means of the simulated per-run differences are thus independent by construction. The calibration
(family-wise false-confirmation rate and CI coverage) is valid for independent partitions. It does not show how the
registered t-test behaves when partitions share test and training data.

## 2. Corrected resampled t-test

Registered test (SAP v3.3): one-sample t-test on the J = 5 partition means, SE² = s²/J, df = J − 1 = 4, where s² is the
variance of the partition means. The corrected test (Nadeau & Bengio, 2003) uses SE² = s² · (1/J + n_test/n_train),
with the same estimate and df. Holm is applied over H1–H7, as registered.

### Sizes used, in cases

The SUN partitions are identical in size (`partition_sizes.csv`):

| Split | Positive cases | Negative virtual cases | Total |
|---|---|---|---|
| Test | 15 | 15 | 30 |
| Val | 15 | 15 | 30 |
| Train | 70 | 70 | 140 |
| E1 train subsets (5/10/25/50/100 %) | 4/7/18/35/70 | 4/7/18/35/70 | 8/14/36/70/140 |

n_train is the training split, without validation. Validation selects the epoch and the threshold but its cases are not
fitted.

| Hypothesis | n_test/n_train | Basis |
|---|---|---|
| H1, H2 (E1, AULC) | **0.7253** | E1 trains at five budgets. The AULC is a weighted sum of the five per-budget differences, so the ratio is the AULC-weighted mean of 30/n_train(p): weights 0.026, 0.105, 0.211, 0.395, 0.263 for 8, 14, 36, 70, 140 cases. Alternative: 30/140 = 0.2143 (full training split), below. |
| H3 (E2) | 0.2143 | sun_full model (140 training cases) evaluated on its partition's test split (30 cases). The whole split is used; the flat stratum alone (20 cases) would give 0.143, a smaller correction. |
| H4 (E3), H5 | 0.2143 | See below. |
| H6, H7 (E4) | 0.2143 | See below. |

**E3 and E4: why the SUN sizes are used.**

- The five replicates differ through the SUN partition that trained the `sun_full` model: 140 training cases, 30 held
  out as test. The evaluation is on PolypGen, outside SUN.
- In E3 the test set (`polypgen_clean_test`) is identical for every partition. In E4 the evaluation splits come from the
  same PolypGen pool.
- The correlation that the correction targets is the one between models trained on overlapping SUN data, so the ratio
  of the SUN resampling, 30/140, is used.
- The variability of the shared PolypGen test set is not in the partition means at all, corrected or not. It is what
  the registered cluster bootstrap measures, and the registered confirmation rule requires both.
- The E4 support sets (H7) add a few-shot adaptation per pack. They are not part of the SUN resampling and are not
  added to the ratio.

### Check

The uncorrected test computed here reproduces the registered values: 42/42 quantities equal, max |diff| 4.1e-11
(`registered_check.csv`). The quantities are estimate, SE, CI, p and Holm p for H1–H7. The registered double rule is
also reproduced (Holm p < 0.05 and the bootstrap CI excluding 0).

### Results

| H | Estimate | Registered: t-test 95 % CI | p | Holm p | Corrected: n_test/n_train | 95 % CI | p | Holm p |
|---|---|---|---|---|---|---|---|---|
| H1 | 0.0475 | [0.0286, 0.0663] | 0.0022 | 0.0088 | 0.7253 | [0.0069, 0.0880] | 0.0314 | 0.1254 |
| H2 | −0.0138 | [−0.0285, 0.0009] | 0.0592 | 0.1185 | 0.7253 | [−0.0455, 0.0178] | 0.2913 | 0.5825 |
| H3 | 0.0380 | [0.0122, 0.0639] | 0.0151 | 0.0452 | 0.2143 | [0.0008, 0.0752] | 0.0470 | 0.1410 |
| H4 | −0.0720 | [−0.0985, −0.0455] | 0.0017 | 0.0084 | 0.2143 | [−0.1102, −0.0338] | 0.0064 | 0.0325 |
| H5 | 0.1548 | [0.1202, 0.1895] | 0.0002 | 0.0017 | 0.2143 | [0.1050, 0.2047] | 0.0010 | 0.0069 |
| H6 | −0.0064 | [−0.0241, 0.0114] | 0.3757 | 0.3757 | 0.2143 | [−0.0318, 0.0191] | 0.5271 | 0.5825 |
| H7 | −0.0748 | [−0.1012, −0.0485] | 0.0014 | 0.0084 | 0.2143 | [−0.1128, −0.0369] | 0.0054 | 0.0325 |

The SE grows by a factor √(1 + J·r): 2.15 for H1 and H2, 1.44 for the others.

**Alternative for E1, ratio 30/140 for H1 and H2** (`corrected_t_alt_e1_full_train.csv`; Holm over H1–H7 again; the
rows H3–H7 are unchanged):

| H | 95 % CI | p | Holm p |
|---|---|---|---|
| H1 | [0.0203, 0.0746] | 0.0083 | 0.0332 |
| H2 | [−0.0350, 0.0073] | 0.1436 | 0.2873 |

With this alternative the Holm p of H3–H7 change only through the ranking: H6 0.5271, the others as in the main table.

### Registered double rule with the corrected p (descriptive)

The rule is Holm p < 0.05 and the cluster-bootstrap CI excluding 0. The bootstrap is unchanged.

| H | Registered bootstrap CI | Registered | Corrected (E1: 0.7253) | Corrected (E1: 0.2143) |
|---|---|---|---|---|
| H1 | [0.0355, 0.0592] | confirmed | not met (Holm 0.125) | met (Holm 0.033) |
| H2 | [−0.0214, −0.0064] | not confirmed | not met | not met |
| H3 | [0.0202, 0.0593] | confirmed | not met (Holm 0.141) | not met (Holm 0.141) |
| H4 | [−0.3447, 0.0944] | not confirmed | not met (bootstrap includes 0) | not met |
| H5 | [−0.0310, 0.3093] | not confirmed | not met (bootstrap includes 0) | not met |
| H6 | [−0.0184, 0.0058] | not confirmed | not met | not met |
| H7 | [−0.1353, −0.0150] | confirmed | met (Holm 0.032) | met (Holm 0.032) |

No sign changes, and no corrected t-test CI crosses zero where the registered one did not. With the correction, H3's
CI nearly touches 0 (lower limit 0.0008) and its Holm p rises above 0.05. H1 depends on the E1 ratio.

## 3. Overlap between partitions

Pairs of test splits (`partition_overlap.csv`; each test split has 15 positive cases, 15 negative virtual cases and
negatives from 3 of the 13 negative videos):

| Pair | Shared positive test cases | Shared negative virtual cases | Shared negative test videos | Test cases of one in the train of the other | Shared train cases (of 140) |
|---|---|---|---|---|---|
| k1–k2 | 1 | 0 | 1 | 12 / 12 | 48 |
| k1–k3 | 3 | 0 | 2 | 7 / 7 | 55 |
| k1–k4 | 0 | 0 | 1 | 13 / 13 | 46 |
| k1–k5 | 3 | 0 | 0 | 10 / 10 | 52 |
| k2–k3 | 2 | 0 | 0 | 10 / 12 | 48 |
| k2–k4 | 3 | 0 | 1 | 10 / 8 | 51 |
| k2–k5 | 1 | 0 | 2 | 10 / 12 | 55 |
| k3–k4 | 4 | 0 | 0 | 10 / 10 | 59 |
| k3–k5 | 1 | 0 | 0 | 12 / 9 | 59 |
| k4–k5 | 2 | 0 | 0 | 11 / 8 | 58 |

- **Positive test cases:** pairs share 0–4 of 15 (20 shared case pairs in total).
- **Negative test cases:** no negative virtual case is shared. Pairs share 0–2 of their 3 negative videos (7 in total),
  with different 33-frame chunks of the same video.
- **Test vs training:** 7–13 of the 30 test cases of one partition are in the training split of another.
- **Training:** pairs share 46–59 of their 140 training cases.

Sensitivity analysis: no verdicts. The registered results and verdicts are unchanged.

## Files

| File | Content |
|---|---|
| `corrected_t.csv` | per hypothesis: estimate, partition-mean SD, ratio and its basis, registered and corrected SE, t, CI, p, Holm p, bootstrap CI, double rule |
| `corrected_t_alt_e1_full_train.csv` | the same with 30/140 for H1 and H2 |
| `partition_sizes.csv`, `partition_overlap.csv` | cases per split and partition; pairwise overlap |
| `registered_check.csv`, `summary.json`, `run_log.txt` | reproduction check and run information |
