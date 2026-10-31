# Sensitivity plan: E4 support size vs type of negatives

## Status

**Post hoc sensitivity analysis, not registered in the SAP, decided on 2026-10-31 after seeing the results** of the
registered analysis (SAP v3.3) and of the common-evaluation-set analysis (`results_v3/verification/e4_common_eval/`,
commit `5b67baa`). **Its results will not be used to confirm or refute hypotheses**: no Holm correction and no
confirmation verdicts; estimates and intervals are reported as observed. The registered analysis and its results do not
change.

## Question

Is the improvement of the adapted arms at 500 frames due to the size of the support, or to the support containing
sequence negatives, of the same type as the negatives of the test?

## Background

On the common evaluation set (the frames of the 500-frame evaluation split, present in the four budgets), the arm
contrasts change sign between 200 and 500 frames. Balanced accuracy, mean of the four regimes:

| Contrast | 200 frames | 500 frames |
|---|---|---|
| C − A | −0.0948 | +0.0035 |
| D − A | −0.0553 | +0.0870 |

The registered supports differ in more than size:

| Registered support | Positives | Isolated negatives | Sequence negatives |
|---|---|---|---|
| 200 frames (`s200`) | 100 (80 train / 20 val) | 100 | 0 |
| 500 frames (`s500`) | 250 | 184 | 66, from 14 sequences |

The test of every pack (the common set = the `s500` evaluation split) holds 1,004 positives and only sequence negatives
(302–1,031 frames, from 4 sequences), because the `s500` support takes some frames of the other negative sequences and
the split keeps sequences disjoint.

## Design

### Packs: `data_packs/v3/polypgen_fewshot_s200seq/`

Fifteen packs (5 partitions × 3 seeds), generated in polyp-db-analysis (commits `56c7a1b` and `fca9597`, script
`scripts/build_polypgen_fewshot_s200seq.py`) and copied from `<DATA_ROOT>/data_packs_v3_staging/polypgen_fewshot_s200seq/`.
Each pack has:

- **Positives:** the same 100 frames as the registered `s200` pack, with the same train/val split (80/20).
- **Negatives:** 100 sequence negatives (80 train / 20 val), drawn from the negative sequences assigned to the support
  of the registered `s500` pack of the same pack: train sequences to train, val sequences to val. Within each split,
  frames are chosen round-robin across sequences, ranked by `hmac_sha256(frame_id, 200000 + pack_seed)`.
- **Test:** the registered `s500` test split, byte for byte.

Checks done before this plan (2026-10-31, nothing trained):

- **Copy:** CSVs byte-identical to the staging packs. Only the manifests changed: real paths were replaced by
  `<DATA_ROOT>`, with the rules of `scripts/v3/release_depersonalize_paths.py`.
- **Hashes:** the SHA-256 of every `train.csv`, `val.csv` and `test.csv` equals the hash in `index.yaml` and in the
  manifest of its pack (15 × 3).
- **Test:** each `test.csv` is byte-identical to the registered `v3/k{k}/polypgen_fewshot/seed{s}/s500/test.csv`.
- **Leakage:** no support frame (train or val) is in the test. The negative sequences of train, val and test are
  pairwise disjoint.
- **Positives:** the positives of train and val equal those of the registered `s200` pack.
- **Negatives:** the train and val negative sequences are subsets of those of the registered `s500` train and val. No
  isolated negative is present in any split.
- **Pack verifier:** `scripts/v3/verify_data_packs.py --v3`, extended to read the `index.yaml` of this pack family,
  passes for all of `data_packs/v3/`: 416 splits, 266 CSVs, no failure. For the new packs it checks hashes, counts,
  the parent pack, and that every image exists with its recorded size and SHA-1 (record
  `results_v3/verification/e4_negtype_packs/`).

Composition per pack (`composition.csv`):

| | Frames | Positives | Sequence negatives | Negative sequences |
|---|---|---|---|---|
| Train | 160 | 80 | 80 | 9–12 |
| Val | 40 | 20 | 20 | **2–5** |
| Test | 1,306–2,035 | 1,004 | 302–1,031 | 4 |

### Runs

- **Adaptation:** 60 runs, the four regimes × 15 packs (`campaign_plan.negtype_matrix`, phase `e4negtype` of
  `run_campaign.py`).
  - Each run adapts the registered `sun_full` model of the same partition, seed and regime
    (`checkpoints_v3/campaign/classification/k{k}/sun_full/...`).
  - It uses the registered 200-frame schedule: `config/exp/v3/e4_budgets/s200seq.yaml` = `s200.yaml` plus the pack
    pattern of `config/data/v3/polypgen_fewshot_s200seq.yaml`.
  - Output under `checkpoints_v3/sens_e4_negtype/classification/k{k}/e4/s200/{regime}_seed{s}/`.
- **Dry run:** the 60 command lines resolve with no SAP-rule failure. Their resolved arguments differ from those of the
  registered `s200` run of the same pack and regime only in the packs, the experiment config name and the output paths
  (`results_v3/verification/campaign_plan_sens_e4_negtype.csv`, `campaign_plan_checks_sens_e4_negtype.json`). The
  registered matrix resolved with the changed code is byte-identical to HEAD (720 runs).
- **Arms:** as in the campaign, the training code writes all four arms of each run:
  - **A:** unadapted model, τ inherited from `sun_full`, on the test. Unchanged: same model and τ as the registered
    runs. It is evaluated on the run's own zero-shot outputs, as in the registered runs. Zero-shot probabilities of the
    same frame differ between registered runs of different budgets by up to 0.009 (inference non-determinism); the
    analysis reports the difference against the registered `s500` zero-shot outputs.
  - **B:** unadapted model; τ re-estimated without training, F1-optimal on the unadapted scores of the new support
    (train + val, 200 frames), as defined in the campaign.
  - **C:** adapted model, inherited τ.
  - **D:** adapted model; τ F1-optimal on the adapted scores of the new support validation split (40 frames). As in the
    campaign, the inherited τ is kept when that split is degenerate.

  Nothing is re-estimated in the analysis.

## Estimands

Three conditions, all evaluated on the common set of each pack:

| Condition | Runs | Evaluation |
|---|---|---|
| `200iso` | registered `s200` (isolated negatives) | restricted to the common set |
| `200seq` | new (sequence negatives) | its test = the common set |
| `500` | registered `s500` | its evaluation split = the common set |

- **Arm contrasts per condition:** B − A, C − A, D − C and D − A, for balanced accuracy, sensitivity, specificity and
  AUROC. AUROC is omitted for B − A and D − C, which are 0 by construction.
- **Key contrasts**, for X = D and X = C, same four metrics:
  - **type of negatives:** [X − A](200seq) − [X − A](200iso);
  - **support size:** [X − A](500) − [X − A](200seq).

Each is a paired difference per run (partition k, seed s), as the mean of the four regimes.

## Inference

The registered code (`analysis/v3`), unchanged:

- **Estimate:** the mean of the 15 per-run values.
- **t-test CI:** one-sample t-test on the 5 partition means (`mixed`, variant `ttest`).
- **Bootstrap CI:** cluster bootstrap of the E4 units (B = 2000, seed 12345). The units are 6 centres for positives and
  24 sequences or centres for negatives. The bootstrap space is anchored on the registered E4 units, so the draws are
  those of the registered e4 family and of `e4_common_eval`. The script checks that the space is identical.

## Fixed reading

The reading uses **balanced accuracy, X = D**, and both 95 % intervals (t-test and bootstrap).

- **Type matters** if both intervals of the type contrast lie above zero.
- **Size matters** if both intervals of the size contrast lie above zero.
- Both can hold at once.
- Intervals entirely below zero are reported as an effect in the opposite direction. Any other case is reported as
  not shown.
- X = C is read with the same rule, as a secondary reading.
- Sensitivity, specificity and AUROC are descriptive.

## Notes on interpretation, declared in advance

- **Validation of D:** the support validation split holds 20 negatives from only 2–5 sequences per pack (and 20
  positives). D's τ is estimated on these 40 frames, so it depends on few sequences. B's τ uses the whole support
  (100 negatives from 14 sequences). The analysis reports, per condition, how many runs kept the inherited τ for D.
- **The size contrast is not size alone.** From `200seq` to `500`, the support changes:
  - positives: 100 → 250;
  - negatives: 100 sequence negatives → 184 isolated + 66 sequence negatives, from the same sequences;
  - schedule: the registered 500-frame schedule (`s500.yaml`).

  It therefore measures everything that differs between the two supports except the presence of sequence negatives.
- **The type contrast** changes only the 100 negatives: isolated → sequence, with the same positives, schedule, parent
  model and test frames.

## Code and its test

`scripts/v3/e4_negtype.py`:

- **Inputs:** it reads the registered campaign and the 200seq runs (`--seq-root`).
- **Run checks:** it verifies the 200seq runs before using them:
  - 60 runs, no others;
  - the campaign integrity checks, except the `sun_full` parent, which lives under the registered root;
  - A and C use the τ of the registered parent;
  - the test is exactly the common set.
- **Reproduction check:** the 200iso and 500 contrasts must equal `results_v3/verification/e4_common_eval/` before
  anything is written.
- **Test before the runs** (`--dry-seq`, 2026-10-31, record `results_v3/verification/e4_negtype_dry_test/`): with the
  registered 200-frame runs standing in for 200seq:
  - 200iso and 500 reproduce `e4_common_eval`: 168/168 quantities equal (estimate, t-test CI, bootstrap CI, B);
  - the type contrasts are exactly 0;
  - the bootstrap space equals the registered one.

## Procedure (phase 2, after review)

1. **Adaptation:** `run_campaign.py e4negtype --root checkpoints_v3/sens_e4_negtype/classification --records
   results_v3/sens_e4_negtype`, on GPUs 0 and 1. The launcher retries transient failures once; any failure is retried
   and reported.
2. **Checks:** for each run, `verify_phase.py e4negtype` with the same arguments, i.e. the section 6 checks of the
   campaign. Then `batch_records.py` writes the records and a hashed manifest, including the checkpoint SHA-256.
3. **Analysis:** `scripts/v3/e4_negtype.py --seq-root checkpoints_v3/sens_e4_negtype/classification --out
   results_v3/verification/e4_negtype`, plus a README.
4. **Clean-up:** delete the adaptation weights (`*.pth` under `checkpoints_v3/sens_e4_negtype/`) after the checks and
   records. The per-frame outputs and metrics JSONs are kept.
5. **Records:** commit the results and a RUNLOG entry; tag `v3-cmpb-sens-e4-negtype-results`; push.

Estimates: registered `s200` adaptations took about 30 s each, so the 60 runs take about 15–20 min on two GPUs. Peak
disk use is about 26 GB before deletion; 277 GB are free.
